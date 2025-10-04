import os
import random
import json
import glob
import numpy as np
from tqdm.auto import tqdm
import wandb
from typing import Optional, List
from functools import partial
from itertools import islice
from dataclasses import dataclass, asdict, field
import tyro

from trl import SFTTrainer, SFTConfig
from datasets import Dataset, IterableDataset

from accelerate.logging import get_logger
from accelerate import Accelerator

from arc25.encoders import create_grid_encoder
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.logging import log_execution_time, configure_logging
from arc25.training_tasks import training_tasks_generator
from arc25.utils import set_random_seed
from arc25.collator import get_data_collator
from arc25.model import get_model, get_tokenizer, get_lora_model

logger = get_logger(__name__)

@dataclass
class Config:
    output_dir: str
    verbose: bool = True
    resume_from_checkpoint: bool = True
    model_path: str = '/home/gbarbadillo/models/Qwen2.5-Coder-0.5B-Instruct/'
    adapter_path: Optional[str] = None
    use_4bit_quantization: bool = False
    # train_datasets: List[List[str]] = field(default_factory=lambda: [['/mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json', 'output-from-examples-v0']])
    # remove_train_samples_to_fit_max_seq_len: bool = False
    # subsample_train_tasks_ratio: Optional[float] = None
    # val_dataset: List[str] = field(default_factory=lambda: ['/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json', 'output-from-examples-v0'])
    n_gpus: int = 2
    device_map: str = 'None' # 'custom', 'balanced', 'auto', 'None'
    max_seq_len: int = 4096
    max_steps : Optional[int] =  6000
    logging_steps: int = 10 #10a
    eval_steps: int = 50 #50
    save_steps: Optional[int] = None
    log_to_wandb: bool = True
    warmup_ratio: float = 0.05
    batch_size: int = 16 #16
    random_seed: Optional[int] = None # None, 7
    val_random_seed: int = 42
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1 # if using 2 the validation loss is not correctly computed
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear" #linear, constant_with_warmup, cosine, cosine_with_restarts
    lr_num_cycles: int = 4 # only applicable for cosine_with_restarts
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit" # "paged_adamw_8bit"
    torch_dtype: str = "bfloat16" # "bfloat16" or "float16", float16 causes divergence when training on my PC, but it is 4x faster on Kaggle
    packing: bool = False # multiple short examples are packed in the same input sequence to increase training efficiency
    use_liger_kernel: bool = False # reduces memory usage by 60% and in theory increase speed by 20%
    dataloader_num_workers: int = 4 # Number of subprocesses to use for data loading, if set to 0, the data will be loaded in the main process
    # LoRA
    use_lora: bool = True
    use_rslora: bool = True
    use_dora: bool = False # Currently it is not supported by VLLM
    lora_r: int = 16
    lora_weight_initialization: str = 'default' # 'gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq', 'default'
    # Data augmentation
    # compose_new_task_probability: float = 0.0
    # compose_new_task_weights: Optional[List[float]] = None
    # Verify
    # verify_correct_output_probability: float = 0.5


@log_execution_time
def fine_tuning_main():
    cfg = tyro.cli(Config)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false' # to avoid warnings, so far I haven't seen any slowdown
    configure_logging()
    save_train_conf(cfg)
    if cfg.log_to_wandb:
        accelerator = Accelerator(log_with='wandb')
        accelerator.init_trackers(
            project_name=os.path.basename(os.path.dirname(cfg.output_dir)),
            config=cfg,
            init_kwargs={"wandb": dict(dir=cfg.output_dir,
                                       name=os.path.basename(cfg.output_dir))}
        )
    else:
        accelerator = Accelerator()
    logger.info(f'Train configuration: {asdict(cfg)}')

    model = get_model(cfg.model_path, torch_dtype=cfg.torch_dtype,
                      use_4bit_quantization=cfg.use_4bit_quantization, device_map=cfg.device_map,
                      use_gradient_checkpointing=cfg.gradient_checkpointing)
    tokenizer = get_tokenizer(cfg.model_path, model, cfg.grid_encoder)
    if cfg.use_lora:
        model = get_lora_model(model, cfg.adapter_path, cfg.lora_r, cfg.use_rslora,
                               cfg.use_dora, cfg.lora_weight_initialization)
    else:
        logger.info('Not using LoRA, full model will be fine-tuned')


    if cfg.random_seed is not None:
        current_process_seed = cfg.random_seed + accelerator.process_index
    else:
        current_process_seed = random.randint(0, 2**32 - 1)

    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    dataset_kwargs = {'grid_encoder': grid_encoder, 'tokenizer': tokenizer}
    train_dataset = IterableDataset.from_generator(
        partial(random_prompt_generator, **dataset_kwargs),
        gen_kwargs={"shard": [current_process_seed + i for i in range(max(cfg.dataloader_num_workers, 1))],
                    "verbose": [True] + [False] * (cfg.dataloader_num_workers - 1)})
    val_generator = random_prompt_generator(grid_encoder, tokenizer, cfg.val_random_seed)
    val_dataset = Dataset.from_dict({'input_ids': [x['input_ids'] for x in islice(val_generator, 100)]})

    # TODO: undo tokenization to show the prompt
    # if accelerator.is_main_process: # Ensure printing only happens once in multi-GPU setups
    #     logger.info("Sampling one element from the training dataset:")
    #     pretty_print_prompt(next(iter(train_dataset))['text'])

    # train_dataset = Dataset.from_dict({'text': [next(iter(train_dataset))['text'] for _ in range(16*200)]})

    training_arguments = get_training_arguments(cfg)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=get_data_collator(tokenizer),
        args=training_arguments,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint and is_checkpoint_available(cfg.output_dir))
    # torch.distributed.destroy_process_group()


def save_train_conf(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'cfg.json'), 'w') as f:
        json.dump({key:value for key, value in cfg.__dict__.items() if not key.startswith('__')}, f, indent=4)

############################################################################
# Data
############################################################################

class PromptTokenDistributionLogger():
    def __init__(self, tokenizer, period=1000):
        self.period = period
        self.tokenizer = tokenizer
        self.prompt_lengths = []

    def add_prompt(self, prompt):
        # TODO: avoid double tokenization
        self.prompt_lengths.append(len(self.tokenizer.encode(prompt)))
        if len(self.prompt_lengths) >= self.period:
            log_prompt_length_percentiles(self.prompt_lengths, 'train')
            self.prompt_lengths = []


def log_prompt_length_percentiles(prompt_lengths, prefix):
    percentiles = [10, 50, 75, 90, 95, 97, 98, 99]
    percentile_to_n_tokens = {percentile: int(np.percentile(prompt_lengths, percentile)) for percentile in percentiles}
    logger.info(f'\t{prefix} number of prompts: {len(prompt_lengths)}, max number of tokens : {max(prompt_lengths)}, percentiles: {percentile_to_n_tokens}')


def random_prompt_generator(grid_encoder, tokenizer, shard, verbose=False):
    shard = shard[0] if isinstance(shard, list) else shard
    verbose = verbose[0] if isinstance(verbose, list) else verbose
    logger.info(f'Starting random prompt generator with shard: {shard}')
    set_random_seed(shard)
    generator = training_tasks_generator(verbose)
    prompt_distribution_logger = PromptTokenDistributionLogger(tokenizer)
    for task in generator:
        prompt_version = 'code-from-examples-v3'
        prompt = create_prompt_from_task(
            task, prompt_version=prompt_version, grid_encoder=grid_encoder, tokenizer=tokenizer)
        if verbose: prompt_distribution_logger.add_prompt(prompt)
        yield {'input_ids': tokenizer.encode(prompt, return_tensors='pt').squeeze(0).tolist()}


############################################################################
# Training
############################################################################

def get_training_arguments(cfg):
    gradient_accumulation_steps = get_gradient_accumulation_steps(
        cfg.batch_size, cfg.per_device_train_batch_size, cfg.n_gpus, cfg.device_map)
    batch_size_kwargs = dict(
        # 4-16 batch size should be fine for lora.
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
    )
    scheduler_type = cfg.lr_scheduler_type
    if scheduler_type == 'cyclic':
        logger.info('Using cyclic learning rate scheduler (renaming to linear because it will be hacked later)')
        scheduler_type = 'linear'

    lr_scheduler_kwargs = {}
    if cfg.lr_scheduler_type == 'cosine_with_restarts':
        lr_scheduler_kwargs['num_cycles'] = cfg.lr_num_cycles
    training_arguments = SFTConfig(
            # https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
            output_dir=cfg.output_dir,
            save_total_limit=3, # I'm only interested in the last checkpoint, I will be saving 3 to avoid corruption problems (2 will be enough for this)
            num_train_epochs=0,
            max_steps=cfg.max_steps,
            warmup_ratio=cfg.warmup_ratio,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=scheduler_type, #constant_with_warmup, cosine, cosine_with_restarts
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            gradient_checkpointing=cfg.gradient_checkpointing,
            optim=cfg.optim,
            max_grad_norm=cfg.max_grad_norm,

            dataset_text_field="text",
            max_seq_length=cfg.max_seq_len,

            do_eval=True if cfg.eval_steps else False,
            eval_strategy='steps' if cfg.eval_steps else 'no', # 'epoch', 'steps', 'no'
            save_steps=cfg.save_steps or cfg.eval_steps,
            logging_steps=cfg.logging_steps, #50,
            eval_steps=cfg.eval_steps,
            log_level="info",
            report_to='wandb' if cfg.log_to_wandb else 'tensorboard',

            # parameters added to make the code work with accelerate
            accelerator_config=dict(
                dispatch_batches=False), # If set to True, the dataloader prepared by the Accelerator is only iterated through on the main process and then the batches are split and broadcast to each process.
            # https://huggingface.co/transformers/v4.9.1/main_classes/trainer.html#trainingarguments
            ddp_find_unused_parameters=False, # only used with accelerate, got a warning saying that it slows down if True

            ignore_data_skip=True, # otherwise it takes too long to start training when resuming from checkpoint
            packing=cfg.packing,
            use_liger_kernel=cfg.use_liger_kernel,

            dataloader_num_workers=cfg.dataloader_num_workers, # Number of subprocesses to use for data loading
            dataloader_pin_memory=True, # Whether you want to pin memory in data loaders or not. Will default to True.
            dataloader_prefetch_factor=4 if cfg.dataloader_num_workers else None, # Number of batches loaded in advance by each worker

            # required by deepspeed
            bf16=True,
            bf16_full_eval=True,

            **batch_size_kwargs
    )
    return training_arguments


def get_gradient_accumulation_steps(batch_size, per_device_train_batch_size, n_gpus, device_map):
    if n_gpus > 1 and device_map == 'None': # multi-gpu accelerate training
        accumulation_steps = batch_size//per_device_train_batch_size//n_gpus
    else:
        accumulation_steps = batch_size//per_device_train_batch_size
    logger.info(f'Using {accumulation_steps} gradient accumulation steps')
    return accumulation_steps


def is_checkpoint_available(output_dir):
    is_checkpoint_available = len(glob.glob(os.path.join(output_dir, 'checkpoint-*'))) > 0
    if is_checkpoint_available:
        logger.info('Checkpoint found, resuming training')
    else:
        logger.info('No checkpoint found, starting training from scratch')
    return is_checkpoint_available


if __name__ == '__main__':
    fine_tuning_main()
