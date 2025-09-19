
from arc25.logging import configure_logging, logging
from arc25.utils import set_cuda_visible_devices_to_least_used_gpu_if_undefined

configure_logging()
set_cuda_visible_devices_to_least_used_gpu_if_undefined()

import tyro
import os
import random
from unsloth import FastLanguageModel
from dataclasses import dataclass
from datasets import Dataset
from tqdm.auto import tqdm
from functools import partial, update_wrapper

from trl import GRPOConfig, GRPOTrainer

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions, convert_task_to_numpy, set_random_seed, is_checkpoint_available
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.logging import configure_logging, logging, log_execution_time
from arc25.parallel_code_execution import CodeRunner

configure_logging()
logger = logging.getLogger(__name__)


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
    # LoRA
    lora_r: int = 16
    use_rslora: bool = True
    # dataset
    dataset_path: str = "/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small_arc-agi_training_challenges.json"
    output_dir: str = "/mnt/hdd0/Kaggle/arc25/trainings/2025-09-15-debug-grpo/lr1e-5_small-dataset_10epochs_5582e5ca"
    # training hyperparameters
    epochs: int = 10
    num_generations: int = 4
    training_prompts_per_step: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'constant_with_warmup'
    use_data_augmentation: bool = True
    resume_from_checkpoint: bool = True
    warmup_ratio: float = 0.1
    # others
    n_jobs: int = -1


def main():
    cfg = tyro.cli(Config, description="Fine-tune a language model on ARC tasks using RL with GRPO and unsloth.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(f"Configuration: {cfg}")

    llm, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_path, load_in_4bit=cfg.load_in_4bit,
        fast_inference=True, max_seq_length=cfg.max_seq_length,
        gpu_memory_utilization=cfg.gpu_memory_utilization)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)

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
            grpo_dataset.append(dict(prompt=prompt, tasks=task))
    grpo_dataset = Dataset.from_list(grpo_dataset)
    pretty_print_prompt(prompt, default_color='white')

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
        per_device_train_batch_size=cfg.num_generations, # this is forced by unsloth
        num_generations=cfg.num_generations,
        gradient_accumulation_steps=cfg.training_prompts_per_step,
        warmup_ratio=cfg.warmup_ratio,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=0.1,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=None,
        temperature=1.0,
        top_p=0.95,
        dataloader_num_workers=1,
        save_steps=100,
        mask_truncated_completions=True, #  When enabled, truncated completions are excluded from the loss calculation, preventing them from being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is a good practice for training stability.
        completion_only_loss=True,
        # wandb
        report_to='wandb',
        run_name=os.path.basename(cfg.output_dir),
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
    logger.info(f'Completions length: {[len(c) for c in completion_ids]}')
    rewards = [_individual_arc_reward(result, task) for result, task in zip(results, tasks)]
    logger.info(f'Rewards: {rewards}')
    return rewards


def _individual_arc_reward(result, task):
    """
    The north start metric is the correct grids, pixel score is use as a tiebreaker.
    When code is not parsed reward is -1, and code that creates valids gets a reward of 1 vs code that does not.
    -1 -> code not parsed
    0 -> code parsed but does not produce valid results
    1 -> code produces valid results but accuracy is 0
    1 + 8*correct_grids + pixel_score -> code produces valid results with accuracy
    """
    if 'code' not in result: # code was not parsed correctly
        reward = -1.0
    elif 'train_correct_grids' not in result: # code ran but did not produce valid results
        reward = 0.0
    else:
        n_train, n_test = len(task['train']), len(task['test'])
        correct_grids = (float(result['train_correct_grids'])*n_train + float(result.get('test_correct_grids', 0))*n_test) / (n_train + n_test)
        pixel_score = (float(result['train_pixel_score'])*n_train + float(result.get('test_pixel_score', 0))*n_test) / (n_train + n_test)
        reward = 1.0 + 8*correct_grids + pixel_score
    return reward


if __name__ == "__main__":
    main()
