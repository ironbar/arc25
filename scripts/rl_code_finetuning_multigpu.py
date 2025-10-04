
from arc25.logging import configure_logging, logging

configure_logging()

import tyro
import os
import random
import numpy as np
from dataclasses import dataclass
from datasets import Dataset
from tqdm.auto import tqdm
from functools import partial, update_wrapper
from accelerate.logging import get_logger
from accelerate import Accelerator

from trl import GRPOConfig, GRPOTrainer

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions, convert_task_to_numpy, set_random_seed, is_checkpoint_available
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.logging import configure_logging, logging, log_execution_time
from arc25.parallel_code_execution import CodeRunner
from arc25.model import get_model, get_tokenizer, get_lora_model

logger = get_logger(__name__)


@dataclass
class Config:
    # base model
    model_path: str = "/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B"
    load_in_4bit: bool = True
    gpu_memory_utilization: float = 0.7
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
    epochs: int = 10
    save_steps: int = 100 # each checkpoint with lora_r=32 takes around 500MB
    num_generations: int = 4
    gradient_accumulation_steps: int = 1 # the number of generations must be divisible by this
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'constant_with_warmup'
    use_data_augmentation: bool = True
    resume_from_checkpoint: bool = True
    warmup_ratio: float = 0.1
    scale_rewards: str = 'group'
    mask_truncated_completions: bool = True
    # others
    n_jobs: int = -1


def main():
    accelerator = Accelerator()
    cfg = tyro.cli(Config, description="Fine-tune a language model on ARC tasks using RL with GRPO and unsloth.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(f"Configuration: {cfg}")

    grid_encoder = create_grid_encoder(cfg.grid_encoder)

    model = get_model(cfg.model_path, torch_dtype="bfloat16",
                      use_4bit_quantization=cfg.load_in_4bit, device_map='None',
                      use_gradient_checkpointing=True)
    tokenizer = get_tokenizer(cfg.model_path, model, cfg.grid_encoder)
    model = get_lora_model(model, None, cfg.lora_r, cfg.use_rslora, False, 'default')

    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())
    print(f"Loaded {len(dataset)} tasks from {cfg.dataset_path}")
    set_random_seed(None)
    grpo_dataset = []
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
                grpo_dataset.append(dict(prompt=prompt, tasks=task))
    grpo_dataset = Dataset.from_list(grpo_dataset)
    pretty_print_prompt(prompt, default_color='white')

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
        max_grad_norm=0.1,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=cfg.max_seq_length - cfg.max_completion_length,
        temperature=1.0,
        top_p=0.95,
        dataloader_num_workers=1,
        save_steps=cfg.save_steps,
        mask_truncated_completions=cfg.mask_truncated_completions, #  When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability.
        scale_rewards=cfg.scale_rewards, # "group", 'batch', 'none', by default is 'group
        repetition_penalty=cfg.repetition_penalty,
        # use_liger_kernel=True, #ImportError: cannot import name '_CONFIG_FOR_DOC' from 'transformers.models.gemma.modeling_gemma' (/home/gbarbadillo/miniconda3/envs/arc25/lib/python3.10/site-packages/transformers/models/gemma/modeling_gemma.py)
        # wandb
        logging_steps=1,
        report_to='wandb',
        run_name=os.path.basename(cfg.output_dir),
        shuffle_dataset=False, # already shuffled on creation
        # vllm
        use_vllm=True,
        vllm_mode="colocate",
        vllm_enable_sleep_mode=False,
        vllm_gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    os.environ["WANDB_PROJECT"] = os.path.basename(os.path.dirname(cfg.output_dir))
    os.environ["WANDB_DIR"] = cfg.output_dir

    code_runner = CodeRunner(n_jobs=cfg.n_jobs)
    reward_func = partial(arc_reward, code_runner=code_runner)
    update_wrapper(reward_func, arc_reward)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=grpo_dataset,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint and is_checkpoint_available(cfg.output_dir))


@log_execution_time
def arc_reward(completions, tasks, completion_ids, code_runner, **kwargs):
    """
    Reward function that rewards completions based on how many test cases they pass.

    As input seems to be receiving: completions, prompts, ground_truth and completion_ids
    """
    numpy_tasks = [convert_task_to_numpy(task) for task in tasks]
    results = code_runner.run(
        numpy_tasks, list(range(len(completions))), completions,
        [None]*len(completions), group_results_by_task=False, disable_tqdm=True)
    completion_lengths = [len(c) for c in completion_ids]
    logger.info(f'Mean completion length: {np.mean(completion_lengths):.2f}, Max completion length: {np.max(completion_lengths):.2f}, lengths: {completion_lengths}')
    rewards = [_individual_arc_reward(result, task) for result, task in zip(results, tasks)]
    logger.info(f'Mean reward: {np.mean(rewards):.2f}, Max reward: {np.max(rewards):.2f}, rewards: {np.array(rewards).round(2).tolist()}')
    logger.info(f'Best completion:\n{completions[np.argmax(rewards)]}')
    return rewards


def _individual_arc_reward(result, task):
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
    # if 'code' not in result: # code was not parsed correctly
    #     reward = -1.0
    # elif 'train_correct_grids' not in result: # code ran but did not produce valid results
    #     reward = 0.0
    if 'train_correct_grids' not in result: # code ran but did not produce valid results
        reward = 0.0
    else:
        n_train, n_test = len(task['train']), len(task['test'])
        correct_grids = (float(result['train_correct_grids'])*n_train + float(result.get('test_correct_grids', 0))*n_test) / (n_train + n_test)
        pixel_score = (float(result['train_pixel_score'])*n_train + float(result.get('test_pixel_score', 0))*n_test) / (n_train + n_test)
        reward = 1.0 + 8*correct_grids + pixel_score
    return reward


if __name__ == "__main__":
    main()
