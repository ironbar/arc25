from arc25.utils import set_cuda_visible_devices_to_least_used_gpu_if_undefined
from arc25.logging import configure_logging

configure_logging()
set_cuda_visible_devices_to_least_used_gpu_if_undefined()

import os
import time
from dataclasses import dataclass
from typing import Optional
import tyro
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from arc25.logging import logging, log_execution_time
from arc25.utils import get_timestamp, load_arc_dataset_with_solutions
from arc25.encoders import create_grid_encoder
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import Template

logger = logging.getLogger(__name__)

@dataclass
class Config:
    output_folder: str
    base_model_path: str = "/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B"
    dataset_path: str = '/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_training_challenges.json'
    lora_path: Optional[str] = None
    use_4bit_quantization: bool = False
    tensor_parallel_size: int = 1
    use_data_augmentation: bool = True
    batch_size: int = 8
    n_predictions: int = 8
    # model parameters
    max_model_len: int = 10700
    max_output_tokens: int = 2048
    gpu_memory_utilization: float = 0.85


def main():
    cfg = tyro.cli(Config, description="Inference with BARC models")
    logger.info(f'Running BARC inference with config: {cfg}')
    llm, tokenizer = load_vllm_model_and_tokenizer(
        cfg.base_model_path, use_4bit_quantization=cfg.use_4bit_quantization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        enable_lora=cfg.lora_path is not None, max_model_len=cfg.max_model_len, max_lora_rank=32,
        gpu_memory_utilization=cfg.gpu_memory_utilization)
    if cfg.lora_path is not None:
        lora_request = LoRARequest(lora_name='LoRA', lora_int_id=1, lora_path=cfg.lora_path)
    else:
        lora_request = None
    grid_encoder = create_grid_encoder('ColorNameEncoder()')
    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())

    sampling_params = SamplingParams(n=cfg.batch_size, temperature=1.0, top_p=0.95, max_tokens=cfg.max_output_tokens)
    os.makedirs(cfg.output_folder, exist_ok=True)
    n_rounds = cfg.n_predictions//cfg.batch_size
    for round_idx in range(n_rounds):
        prompts, data_augmentation_params = [], []
        for task_id in task_ids:
            task = dataset[task_id]
            if cfg.use_data_augmentation:
                params = get_random_data_augmentation_params()
                task = apply_data_augmentation(task, **params)
            else:
                params = None
            data_augmentation_params.append(params)
            prompt = create_prompt_from_task(
                task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
            prompts.append(prompt)

        t0 = time.time()
        text_predictions = llm.generate(prompts, sampling_params, lora_request=lora_request)
        total_tokens = sum(sum(len(_output.token_ids) for _output in output.outputs) for output in text_predictions)
        inference_time = time.time() - t0
        logger.info(f'Prediction round {round_idx + 1}/{n_rounds} completed.')
        logger.info(f"Total tokens generated: {total_tokens}")
        logger.info(f"Time taken: {inference_time:.2f} seconds")
        logger.info(f"Average time per task: {inference_time / len(text_predictions):.2f} seconds")
        logger.info(f"Average tokens per task: {total_tokens / len(text_predictions) / sampling_params.n:.2f} tokens")
        logger.info(f"Average tokens per second: {total_tokens / inference_time:.2f} tokens/second")

        predictions = dict()
        for task_id, output, params in zip(task_ids, text_predictions, data_augmentation_params):
            predictions[task_id] = {
                'text_predictions': [output.text for output in output.outputs],
                'data_augmentation_params': params,
            }

        output_filepath = f'{cfg.output_folder}/{sampling_params.n}preds_{get_timestamp()}_predictions.json'
        with open(output_filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Predictions saved to {output_filepath}")


@log_execution_time
def load_vllm_model_and_tokenizer(model_path: str, use_4bit_quantization: bool=False, tensor_parallel_size: int=1,
               max_model_len: Optional[int]=None, enable_lora: bool=False, max_lora_rank: int=16,
               gpu_memory_utilization: float=0.92):
    logger.info(f"Loading model from {model_path}")
    additional_kwargs = {}
    if max_model_len is not None:
        additional_kwargs['max_model_len'] = max_model_len
    if enable_lora:
        additional_kwargs['max_lora_rank'] = max_lora_rank
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,  # Use less GPU memory
        trust_remote_code=True,
        dtype="bfloat16",  # Use float16 to save memory
        tensor_parallel_size=tensor_parallel_size,  # Single GPU
        quantization="bitsandbytes" if use_4bit_quantization else None,
        enable_prefix_caching=True, # Seems that it is true by default, but let's be explicit
        enable_lora=enable_lora,
        **additional_kwargs
    )
    if model_path.endswith('.gguf'):
        tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer')
    else:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return llm, tokenizer


# TODO: maybe move this to the prompting module
# https://huggingface.co/barc0/Llama-3.1-ARC-Potpourri-Induction-8B
system_prompt = """You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."""

prompt_template_text = """Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:
{% for sample in train_samples %}Example {{ loop.index }}
Input:
{{ sample.input }}

Output:
{{ sample.output }}

{% endfor %}
Here is the input grid for the test example:
{{ test }}

Write a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples.
"""

# I have verified that all responses start with this prefix
common_prefix = "Let's solve this puzzle using Python code with the common library functions. We'll first reason about the problem and then write the code to solve it. The `transform` function will take the input grid and return the output grid. Here is the Python code with the comments describing how to solve the problem:\n" #```python\nfrom common import *\n"

prompt_template = Template(prompt_template_text)

def create_prompt_from_task(task, grid_encoder, tokenizer, shuffle_train_samples=True):
    train_samples = [{'input': grid_encoder.to_text(sample['input']), 'output': grid_encoder.to_text(sample['output'])} for sample in task['train']]
    if shuffle_train_samples:
        random.shuffle(train_samples)
    test_sample = random.choice(task['test'])
    render_kwargs = dict(train_samples=train_samples, test=grid_encoder.to_text(test_sample['input']))
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_template.render(**render_kwargs)},
                {"role": "assistant", "content": common_prefix}]
    prompt = tokenizer.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=False,
                                            continue_final_message=True,
                                            # enable_thinking=False,
                                            )
    return prompt


if __name__ == '__main__':
    main()
