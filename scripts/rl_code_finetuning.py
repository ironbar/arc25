
from arc25.logging import configure_logging, logging
from arc25.utils import set_cuda_visible_devices_to_least_used_gpu_if_undefined

configure_logging()
set_cuda_visible_devices_to_least_used_gpu_if_undefined()

from unsloth import FastLanguageModel
import tyro
import os
import random
import numpy as np
from dataclasses import dataclass
from datasets import Dataset
from tqdm.auto import tqdm
from functools import partial, update_wrapper
from typing import Optional

from trl import GRPOConfig, GRPOTrainer

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions, convert_task_to_numpy, set_random_seed, is_checkpoint_available
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.logging import configure_logging, logging, log_execution_time
from arc25.parallel_code_execution import CodeRunner
from arc25.ngram import ngram_stats

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class Config:
    # base model
    model_path: str = "/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B"
    load_in_4bit: bool = True
    gpu_memory_utilization: float = 0.7
    dtype: Optional[str] = None  # 'float16', 'bfloat16', 'float32', None (If None it uses bfloat16 if possible)
    # max_seq_length: int = 8400+1024  # longest prompt is 8635 tokens, so with 1000 tokens generation it is 9635
    max_seq_length: int = 2048
    max_completion_length: int = 1024
    grid_encoder: str = 'ColorNameEncoder()'
    repetition_penalty: float = 1.0 # 1.0 means no penalty, >1.0 penalizes repetitions
    # LoRA
    lora_r: int = 16
    use_rslora: bool = True
    # dataset
    dataset_path: str = "/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small_arc-agi_training_challenges.json"
    output_dir: str = "/mnt/hdd0/Kaggle/arc25/trainings/2025-09-15-debug-grpo/lr1e-5_small-dataset_10epochs_5582e5ca"
    # training hyperparameters
    reward_name: str = 'arc-v1' # 'arc-v1', 'arc-v2-no-pixel-score'
    epochs: int = 10
    save_steps: int = 100 # each checkpoint with lora_r=32 takes around 500MB
    num_generations: int = 4
    gradient_accumulation_steps: int = 1 # the number of generations must be divisible by this
    max_grad_norm: float = 0.1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'constant_with_warmup'
    use_data_augmentation: bool = True
    resume_from_checkpoint: bool = True
    warmup_ratio: float = 0.1
    scale_rewards: str = 'group'
    mask_truncated_completions: bool = True
    beta: float = 0.001 # KL penalty, by default 0.001 in unsloth
    logging_steps: int = 10 # by default 1 in unsloth
    # others
    n_jobs: int = -1
    code_execution_memory_limit_mb: int = 4096


def main():
    cfg = tyro.cli(Config, description="Fine-tune a language model on ARC tasks using RL with GRPO and unsloth.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(f"Configuration: {cfg}")

    llm, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_path, load_in_4bit=cfg.load_in_4bit,
        fast_inference=True, max_seq_length=cfg.max_seq_length,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype=cfg.dtype)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())
    print(f"Loaded {len(dataset)} tasks from {cfg.dataset_path}")
    set_random_seed(None)

    def dataset_generator():
        for _ in tqdm(range(cfg.epochs), desc="Preparing training data"):
            random.shuffle(task_ids)
            for task_id in list(task_ids):
                if cfg.use_data_augmentation:
                    params = get_random_data_augmentation_params()
                    task = apply_data_augmentation(dataset[task_id], **params)
                else:
                    task = dataset[task_id] # debug without data augmentation
                prompt = create_prompt_from_task(
                        task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
                for _ in range(cfg.gradient_accumulation_steps):
                    yield dict(prompt=prompt, tasks=task)

    grpo_dataset = Dataset.from_generator(dataset_generator)
    pretty_print_prompt(grpo_dataset[0]['prompt'], default_color='white')

    model = FastLanguageModel.get_peft_model(
        llm,
        r = cfg.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = 64,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        use_rslora = cfg.use_rslora,
        # random_state = 3407,
    )

    # https://huggingface.co/docs/trl/main/en/grpo_trainer#trl.GRPOConfig
    training_args = GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=1,
        # unsloth forces num_generations and per_device_train_batch_size to be equal
        per_device_train_batch_size=cfg.num_generations//cfg.gradient_accumulation_steps,
        num_generations=cfg.num_generations//cfg.gradient_accumulation_steps,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=None,
        temperature=1.0,
        top_p=0.95,
        dataloader_num_workers=1,
        save_steps=cfg.save_steps,
        mask_truncated_completions=cfg.mask_truncated_completions, #  When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability.
        scale_rewards=cfg.scale_rewards, # "group", 'batch', 'none', by default is 'group
        repetition_penalty=cfg.repetition_penalty,
        beta=cfg.beta,
        logging_steps=cfg.logging_steps,
        # wandb
        report_to='wandb',
        run_name=os.path.basename(cfg.output_dir),
        shuffle_dataset=False, # already shuffled on creation
    )
    os.environ["WANDB_PROJECT"] = os.path.basename(os.path.dirname(cfg.output_dir))
    os.environ["WANDB_DIR"] = cfg.output_dir

    reward_logger = RewardLogger(n_jobs=cfg.n_jobs, max_completion_length=cfg.max_completion_length,
                                 reward_name=cfg.reward_name, memory_limit_mb=cfg.code_execution_memory_limit_mb)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_logger.arc_reward,
        args=training_args,
        train_dataset=grpo_dataset,
    )
    reward_logger.update_trainer(trainer)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint and is_checkpoint_available(cfg.output_dir))


class RewardLogger():
    """ Computes the reward and adds more logs to the trainer """
    def __init__(self, n_jobs=-1, max_completion_length=1024,
                 reward_name='arc-v1', memory_limit_mb: int = 4096):
        self.code_runner = CodeRunner(n_jobs=n_jobs, memory_limit_mb=memory_limit_mb)
        self.max_completion_length = max_completion_length
        self.trainer = None
        self.reward_name = reward_name


    @log_execution_time
    def arc_reward(self, completions, tasks, completion_ids, **kwargs):
        """
        Reward function that rewards completions based on how many test cases they pass.

        As input seems to be receiving: completions, prompts, ground_truth and completion_ids
        """
        numpy_tasks = [convert_task_to_numpy(task) for task in tasks]
        results = self.code_runner.run(
            numpy_tasks, list(range(len(completions))), completions,
            [None]*len(completions), group_results_by_task=False, disable_tqdm=True)
        rewards = [_individual_arc_reward(result, task, self.reward_name) for result, task in zip(results, tasks)]
        self.log(rewards, completions, completion_ids, results)
        return rewards

    def update_trainer(self, trainer):
        self.trainer = trainer

    def log(self, rewards, completions, completion_ids, results):
        completion_lengths = [len(c) for c in completion_ids]
        # on a first step log to the terminal
        logger.info(f'Mean completion length: {np.mean(completion_lengths):.2f}, Max completion length: {np.max(completion_lengths):.2f}, lengths: {completion_lengths}')
        logger.info(f'Mean reward: {np.mean(rewards):.2f}, Max reward: {np.max(rewards):.2f}, rewards: {np.array(rewards).round(2).tolist()}')
        logger.info(f'Best completion:\n{completions[np.argmax(rewards)]}')
        truncated_completion_ids = [i for i, l in enumerate(completion_lengths) if l >= self.max_completion_length]
        # Log about truncated completions to diagnose collapsing training problem
        if len(truncated_completion_ids) > 0:
            truncated_completion_rewards = [rewards[i] for i in truncated_completion_ids]
            non_truncated_completion_rewards = [rewards[i] for i in range(len(completions)) if i not in truncated_completion_ids]
            logger.warning(f'{len(truncated_completion_ids)}/{len(completions)} completions were truncated to {self.max_completion_length} tokens. Rewards: {truncated_completion_rewards}')
            logger.warning(f'Non-truncated completions rewards: {non_truncated_completion_rewards}')
            logger.warning(f'First truncated completion:\n{completions[truncated_completion_ids[0]]}')
        # second log to the trainer metrics
        if self.trainer is not None:
            metrics = self.trainer._metrics['train']
            metrics["reward_max"].append(float(np.max(rewards)))
            metrics["reward_min"].append(float(np.min(rewards)))
            metrics["truncated_completions_ratio"].append(len(truncated_completion_ids) / len(completions))
            # solved task metrics
            solved_tasks_ids = [i for i, r in enumerate(rewards) if r == 10.0]
            metrics["solved_tasks_ratio"].append(len(solved_tasks_ids) / len(completions))
            if len(solved_tasks_ids) > 0:
                metrics["solved_tasks_completion_length_mean"].append(float(np.mean([completion_lengths[i] for i in solved_tasks_ids])))
                metrics["solved_tasks_completion_length_max"].append(float(np.max([completion_lengths[i] for i in solved_tasks_ids])))
                metrics["is_solved_task"].append(1.0)
            else:
                metrics["is_solved_task"].append(0.0)
            # truncated completions metrics
            if len(truncated_completion_ids) > 0:
                metrics["truncated_completions_reward_mean"].append(float(np.mean([rewards[i] for i in truncated_completion_ids])))
                metrics["truncated_completions_reward_max"].append(float(np.max([rewards[i] for i in truncated_completion_ids])))
                # n gram metrics
                for i in truncated_completion_ids:
                    metrics['truncated_completions_unique_tokens_ratio'].append(len(set(completion_ids[i])) / len(completion_ids[i]) if len(completion_ids[i]) > 0 else 0)
                    for n in [3, 4]:
                        stats = ngram_stats(completion_ids[i], n)
                        metrics[f"truncated_completions_ngram_{n}_unique_ngram_ratio"].append(stats["unique_ngram_ratio"])
                        metrics[f"truncated_completions_ngram_{n}_most_repeated_ngram_count"].append(stats["most_repeated_ngram_count"])
                        metrics[f"truncated_completions_ngram_{n}_most_repeated_ngram_frequency"].append(stats["most_repeated_ngram_frequency"])
            # non truncated completions metrics
            if len(truncated_completion_ids) < len(completions):
                non_truncated_completion_ids = [i for i in range(len(completions)) if i not in truncated_completion_ids]
                metrics["non_truncated_completions_reward_mean"].append(float(np.mean([rewards[i] for i in non_truncated_completion_ids])))
                metrics["non_truncated_completions_reward_max"].append(float(np.max([rewards[i] for i in non_truncated_completion_ids])))
                # n gram metrics
                for i in non_truncated_completion_ids:
                    metrics['non_truncated_completions_unique_tokens_ratio'].append(len(set(completion_ids[i])) / len(completion_ids[i]) if len(completion_ids[i]) > 0 else 0)
                    for n in [3, 4]:
                        stats = ngram_stats(completion_ids[i], n)
                        metrics[f"non_truncated_completions_ngram_{n}_unique_ngram_ratio"].append(stats["unique_ngram_ratio"])
                        metrics[f"non_truncated_completions_ngram_{n}_most_repeated_ngram_count"].append(stats["most_repeated_ngram_count"])
                        metrics[f"non_truncated_completions_ngram_{n}_most_repeated_ngram_frequency"].append(stats["most_repeated_ngram_frequency"])
            # memory error metrics
            memory_errors = [1 for result in results if result.get('error_type', None) == 'MemoryError']
            metrics["memory_errors_ratio"].append(sum(memory_errors) / len(completions))


def _individual_arc_reward(result, task, reward_name):
    """
    The north start metric is the correct grids, pixel score is use as a tiebreaker.

    Reward scheme:
    0: code not parsed or parsed but does not produce valid results
    1: code produces valid results but accuracy is 0
    1 + 8*correct_grids + pixel_score: code produces valid results with accuracy

    Reward is in range [0, 10]

    We should be using a factor of 12 instead of 8 to always give more importance to correct grids than pixel score, because the maximum number of samples per task is 12. However there are only
    20 tasks with more than 8 training samples, and a reward with a maximum value of 10 is more
    intuitive.
    """
    if reward_name == 'arc-v1':
        if 'train_correct_grids' not in result: # code ran but did not produce valid results
            reward = 0.0
        else:
            n_train, n_test = len(task['train']), len(task['test'])
            correct_grids = (float(result['train_correct_grids'])*n_train + float(result.get('test_correct_grids', 0))*n_test) / (n_train + n_test)
            pixel_score = (float(result['train_pixel_score'])*n_train + float(result.get('test_pixel_score', 0))*n_test) / (n_train + n_test)
            reward = 1.0 + 8*correct_grids + pixel_score
        return reward
    if reward_name == 'arc-v2-no-pixel-score':
        if 'train_correct_grids' not in result: # code ran but did not produce valid results
            reward = 0.0
        else:
            n_train, n_test = len(task['train']), len(task['test'])
            correct_grids = (float(result['train_correct_grids'])*n_train + float(result.get('test_correct_grids', 0))*n_test) / (n_train + n_test)
            reward = 1.0 + 9*correct_grids
        return reward
    else:
        raise ValueError(f"Unknown reward name: {reward_name}")


if __name__ == "__main__":
    main()
