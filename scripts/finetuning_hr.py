"""
Fine-tuning with hindsight relabeled tasks.
A minor modification of finetuning.py to use the already saved hindsight relabeled tasks.
"""
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset, IterableDataset

from accelerate.logging import get_logger
from accelerate import Accelerator

from arc25.logging import log_execution_time, configure_logging
from arc25.utils import set_random_seed

from finetuning import (
    get_model, get_tokenizer, get_lora_model, get_data_collator,
    save_train_conf, is_checkpoint_available, get_training_arguments,
    PromptTokenDistributionLogger
)


logger = get_logger(__name__)

@dataclass
class Config:
    output_dir: str
    verbose: bool = True
    resume_from_checkpoint: bool = True
    model_path: str = '/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B'
    adapter_path: Optional[str] = None
    use_4bit_quantization: bool = True
    train_dataset_path: str = '/mnt/hdd0/Kaggle/arc25/data/hindsight_relabeled/2025-08-25_evaluation-85640.json'
    # train_datasets: List[List[str]] = field(default_factory=lambda: [['/mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json', 'output-from-examples-v0']])
    # remove_train_samples_to_fit_max_seq_len: bool = False
    # subsample_train_tasks_ratio: Optional[float] = None
    # val_dataset: List[str] = field(default_factory=lambda: ['/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json', 'output-from-examples-v0'])
    n_gpus: int = 2
    device_map: str = 'None' # 'custom', 'balanced', 'auto', 'None'
    max_seq_len: int = 4096
    max_steps : Optional[int] =  6000
    logging_steps: int = 10 #10a
    eval_steps: int = 0 # Set it to 0 to disable evaluation during training
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
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear" #linear, constant_with_warmup, cosine, cosine_with_restarts
    lr_num_cycles: int = 4 # only applicable for cosine_with_restarts
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit" # "paged_adamw_8bit"
    torch_dtype: str = "bfloat16" # "bfloat16" or "float16", float16 causes divergence when training on my PC, but it is 4x faster on Kaggle
    packing: bool = False # multiple short examples are packed in the same input sequence to increase training efficiency
    use_liger_kernel: bool = True # reduces memory usage by 60% and in theory increase speed by 20%
    dataloader_num_workers: int = 0 # Number of subprocesses to use for data loading, if set to 0, the data will be loaded in the main process
    # LoRA
    use_lora: bool = True
    use_rslora: bool = True
    use_dora: bool = False # Currently it is not supported by VLLM
    lora_r: int = 16
    lora_weight_initialization: str = 'default' # 'gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq', 'default'


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
    tokenizer = get_tokenizer(cfg.model_path, model)
    if cfg.use_lora:
        model = get_lora_model(model, cfg.adapter_path, cfg.lora_r, cfg.use_rslora,
                               cfg.use_dora, cfg.lora_weight_initialization)
    else:
        logger.info('Not using LoRA, full model will be fine-tuned')


    if cfg.random_seed is not None:
        current_process_seed = cfg.random_seed + accelerator.process_index
    else:
        current_process_seed = random.randint(0, 2**32 - 1)

    dataset_kwargs = {'dataset_filepath': cfg.train_dataset_path, 'tokenizer': tokenizer, 'max_seq_len': cfg.max_seq_len}
    train_dataset = IterableDataset.from_generator(
        partial(random_prompt_generator, **dataset_kwargs),
        gen_kwargs={"shard": [current_process_seed + i for i in range(max(cfg.dataloader_num_workers, 1))],
                    "verbose": [True] + [False] * (cfg.dataloader_num_workers - 1)})


    training_arguments = get_training_arguments(cfg)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        data_collator=get_data_collator(tokenizer),
        args=training_arguments,
    )
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint and is_checkpoint_available(cfg.output_dir))
    torch.distributed.destroy_process_group()


def random_prompt_generator(dataset_filepath, tokenizer, max_seq_len, shard, verbose=False):
    shard = shard[0] if isinstance(shard, list) else shard
    verbose = verbose[0] if isinstance(verbose, list) else verbose
    prompt_distribution_logger = PromptTokenDistributionLogger(tokenizer) if verbose else None
    logger.info(f'Starting random prompt generator with shard: {shard}')
    set_random_seed(shard)

    with open(dataset_filepath, 'r') as f:
        tasks = json.load(f)
    task_ids = list(tasks.keys())
    random.shuffle(task_ids)
    prompt_idx = random.randint(0, 1000)
    while True:
        for task_id in task_ids:
            hr_tasks = tasks[task_id]
            prompt = hr_tasks[prompt_idx % len(hr_tasks)]
            # TODO: better implement this
            if prompt_distribution_logger is not None: prompt_distribution_logger.add_prompt(prompt)
            if len(tokenizer.encode(prompt)) > max_seq_len:
                continue
            yield {'text': prompt}
            # yield {'input_ids': tokenizer.encode(prompt, return_tensors='pt').squeeze(0).tolist()}
        prompt_idx += 1

if __name__ == '__main__':
    fine_tuning_main()


