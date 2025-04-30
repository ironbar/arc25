import os
import random
import json
import glob
import numpy as np
from tqdm.auto import tqdm
import wandb
from typing import Optional, List
import argparse
import traceback
from functools import partial
from dataclasses import dataclass, asdict, field
import tyro

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from datasets import Dataset, IterableDataset

# from arc24.encoders import create_grid_encoder
# from arc24.data_augmentation import (
#     random_augment_task,
#     set_random_seed,
#     random_compose_new_task_by_adding_additional_transformation
# )
# from arc24.prompting import create_prompts_from_task, print_smallest_prompt, pretty_print_prompt
# from arc24.data import load_arc_data_with_solutions, BarcDataset
from arc25.logging import log_execution_time, configure_logging

from accelerate.logging import get_logger
from accelerate import Accelerator

logger = get_logger(__name__)

@dataclass
class Config:
    output_dir: str
    verbose: bool = True
    resume_from_checkpoint: bool = True
    model_path: str = '/home/gbarbadillo/models/Qwen2.5-Coder-0.5B-Instruct/'
    adapter_path: Optional[str] = None
    use_4bit_quantization: bool = False
    train_datasets: List[List[str]] = field(default_factory=lambda: [['/mnt/hdd0/Kaggle/arc24/data/new_partitions/train_rs7.json', 'output-from-examples-v0']])
    remove_train_samples_to_fit_max_seq_len: bool = False
    subsample_train_tasks_ratio: Optional[float] = None
    val_dataset: List[str] = field(default_factory=lambda: ['/mnt/hdd0/Kaggle/arc24/data/new_partitions/val_rs7.json', 'output-from-examples-v0'])
    n_gpus: int = 2
    device_map: str = 'None' # 'custom', 'balanced', 'auto', 'None'
    max_seq_len: int = 4096
    epochs = 0
    max_steps : Optional[int] =  6000
    logging_steps: int = 10 #10a
    eval_steps: int = 50 #50
    save_steps: Optional[int] = None
    log_to_wandb: bool = True
    warmup_ratio: float = 0.05
    batch_size: int = 16 #16
    random_seed: Optional[int] = None # None, 7
    grid_encoder: str = 'GridShapeEncoder(RowNumberEncoder(MinimalGridEncoder()))'
    # SmolLM-135M-Instruct: (4, 4); Qwen/Qwen2-0.5B-Instruct: (1, 2)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size = 1 # if using 2 the validation loss is not correctly computed
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "linear" #linear, constant_with_warmup, cosine, cosine_with_restarts
    lr_num_cycles: int = 4 # only applicable for cosine_with_restarts
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit" # "paged_adamw_8bit"
    torch_dtype: str = "bfloat16" # "bfloat16" or "float16", float16 causes divergence when training on my PC, but it is 4x faster on Kaggle
    # LoRA
    use_lora: bool = True
    use_rslora = True,
    use_dora = True,
    lora_r: int = 32
    lora_weight_initialization: str = 'default' # 'gaussian', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'loftq', 'default'
    # Data augmentation
    compose_new_task_probability: float = 0.0
    compose_new_task_weights: Optional[List[float]] = None
    # Verify
    verify_correct_output_probability: float = 0.5


@log_execution_time
def fine_tuning_main():
    cfg = tyro.cli(Config)
    configure_logging()
    save_train_conf(cfg)
    if cfg.log_to_wandb:
        accelerator = Accelerator(log_with=cfg.report_to)
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
                      use_4bit_quantization=cfg.use_4bit_quantization, device_map=cfg.device_map)
    tokenizer = get_tokenizer(cfg.model_path, model)


def save_train_conf(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, 'cfg.json'), 'w') as f:
        json.dump({key:value for key, value in cfg.__dict__.items() if not key.startswith('__')}, f, indent=4)


def get_model(model_path, torch_dtype, device_map, use_4bit_quantization=False):
    logger.info('Loading model...')
    if use_4bit_quantization:
        logger.info('Using 4-bit quantization')
        bnb_config = BitsAndBytesConfig(
            load_in_4bit= True,
            bnb_4bit_quant_type= "nf4",
            bnb_4bit_compute_dtype= torch.float16,
            bnb_4bit_use_double_quant= True,
            llm_int8_enable_fp32_cpu_offload= True,
            llm_int8_skip_modules=['gate', 'lm_head'],
        )
    else:
        bnb_config = None
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=get_device_map(device_map),
        # max_memory={0: '9GB', 1: '8GB'},
        trust_remote_code=True,
        torch_dtype=get_torch_dtype(torch_dtype), #bfloat16 is 4 times slower on Kaggle than float16, on my computer they are the same speed
        attn_implementation=get_flash_attention_implementation(),
        )
    # print(model.hf_device_map)
    print_gpu_memory()
    if use_4bit_quantization:
        # QLoRA on Kaggle is 4 times slower than LoRA, I'm trying to disable gradient checkpointing
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return model


def get_device_map(device_map):
    if device_map == 'None':
        logger.info('Using None device map')
        device_map = None
    elif device_map in ['balanced', 'auto']:
        logger.info(f'Using {device_map} device map')
    else:
        raise ValueError(f'Unknown device map {device_map}')
    return device_map


def get_torch_dtype(torch_dtype):
    if torch_dtype == 'float16':
        logger.info('Using float16 torch dtype')
        return torch.float16
    elif torch_dtype == 'bfloat16':
        logger.info('Using bfloat16 torch dtype')
        return torch.bfloat16
    else:
        raise ValueError(f'Unknown torch dtype {torch_dtype}')


def get_flash_attention_implementation():
    try:
        import flash_attn
        attn_implementation = "flash_attention_2"
    except ImportError:
        attn_implementation = None
    logger.info(f'Using {attn_implementation} attention implementation')
    return attn_implementation


def print_gpu_memory():
    for device in range(torch.cuda.device_count()):
        logger.info(f'GPU {device} memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.1f} GB, max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.1f} GB')


def get_tokenizer(model_path, model, pad_token='<|pad|>'):
    logger.info('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True)
    if tokenizer.pad_token == tokenizer.eos_token:
        if 'qwen' in model_path.lower():
            logger.info('Changing eos token to <|im_end|> for Qwen models, because it is the same as padding token <|endoftext|>')
            tokenizer.eos_token = '<|im_end|>'
        elif 'smollm' in model_path.lower():
            logger.info('Changing pad token to "<|endoftext|>" for SmolLM models, because it is the same as eos token <|im_end|>')
            tokenizer.pad_token = "<|endoftext|>"
        else:
            raise NotImplementedError('Changing padding token is only implemented for Qwen models')
    elif 'pad_token' not in tokenizer.special_tokens_map or tokenizer.pad_token == tokenizer.eos_token:
        logger.info('Adding padding token because the tokenizer does not have one')
        assert pad_token not in tokenizer.get_vocab()
        tokenizer.add_special_tokens({'pad_token': pad_token})
        tokenizer.padding_side = 'right'
        model.resize_token_embeddings(len(tokenizer))
    # if tokenizer.chat_template is None:
    #     logger.warning('The tokenizer does not have a chat template, assigning Qwen2 chat template')
    #     tokenizer.chat_template = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True).chat_template
    #     # tried adding this additional code without success
    #     tokenizer.add_special_tokens({'eos_token': '<|im_end|>'})
    #     tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    #     # tokenizer.eos_token = '<|im_end|>'
    #     # tokenizer.pad_token = '<|endoftext|>'
    assert tokenizer.pad_token != tokenizer.eos_token
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    return tokenizer


if __name__ == '__main__':
    fine_tuning_main()
    
