
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

from trl import GRPOConfig, GRPOTrainer

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions, convert_task_to_numpy, set_random_seed
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.logging import configure_logging, logging
from arc25.parallel_code_execution import run_code_from_predictions

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
        warmup_ratio = 0.1,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm = 0.1,
        max_completion_length=cfg.max_completion_length,
        max_prompt_length=None,
        temperature=1.0,
        top_p=0.95,
        # wandb
        report_to='wandb',
        run_name=os.path.basename(cfg.output_dir),
        # project=os.path.basename(os.path.dirname(cfg.output_dir)),
    )
    os.environ["WANDB_PROJECT"] = os.path.basename(os.path.dirname(cfg.output_dir))
    os.environ["WANDB_DIR"] = cfg.output_dir


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=arc_reward, #reward_num_unique_letters,
        # data_collator=get_data_collator(tokenizer),
        args=training_args,
        train_dataset=grpo_dataset,
        completion_only_loss=True,
    )
    trainer.train()


def arc_reward(completions, tasks, completion_ids, **kwargs):
    """
    Reward function that rewards completions based on how many test cases they pass.

    As input seems to be receiving: completions, prompts, ground_truth and completion_ids
    """
    numpy_tasks = [convert_task_to_numpy(task) for task in tasks]
    results = run_code_from_predictions(numpy_tasks, list(range(len(completions))), completions,
                                        [None]*len(completions), group_results_by_task=False,
                                        disable_tqdm=True)
    logger.info(f'Completions length: {[len(c) for c in completion_ids]}')

    rewards = []
    for result in results:
        if 'code' not in result: # code was not parsed correctly
            rewards.append(-1.0)
        elif 'train_correct_grids' not in result: # code ran but did not produce valid results
            rewards.append(0.0)
        else:
            # TODO: use a smoother reward
            rewards.append(float(result['train_correct_grids']) + float(result.get('test_correct_grids')))

    logger.info(f'Rewards: {rewards}')
    return rewards


if __name__ == "__main__":
    main()
