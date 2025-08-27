import os
from arc25.utils import get_least_used_gpu_index
from arc25.logging import logging, configure_logging, log_execution_time

configure_logging()
os.environ['CUDA_VISIBLE_DEVICES'] = str(get_least_used_gpu_index())

import time
from dataclasses import dataclass
from typing import Optional
import tyro
import json

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from arc25.logging import logging, configure_logging, log_execution_time
from arc25.utils import get_timestamp, load_arc_dataset_with_solutions

logger = logging.getLogger(__name__)

@dataclass
class Config:
    base_mode_path: str
    dataset_path: str
    lora_path: Optional[str] = None
    use_4bit_quantization: bool = False
    tensor_parallel_size: int = 1
    use_data_augmentation: bool = True
    batch_size: int = 8
    n_predictions: int = 8


def main():
    cfg = tyro.cli(Config, description="Inference with BARC models")
    llm, tokenizer = load_vllm_model_and_tokenizer(
        cfg.base_mode_path, use_4bit_quantization=cfg.use_4bit_quantization,
        tensor_parallel_size=cfg.tensor_parallel_size,
        enable_lora=cfg.lora_path is not None, max_model_len=16000, max_lora_rank=32)
    if cfg.lora_path is not None:
        lora_request = LoRARequest('LoRA', 1, adapter_path=cfg.lora_path)
    else:
        lora_request = None
    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())

    sampling_params = SamplingParams(n=cfg.batch_size, temperature=1.0, top_p=0.95, max_tokens=2048)
    for _ in [cfg.n_predictions]:
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
        print(f"Total tokens generated: {total_tokens}")
        print(f"Time taken: {inference_time:.2f} seconds")
        print(f"Average time per task: {inference_time / len(text_predictions):.2f} seconds")
        print(f"Average tokens per task: {total_tokens / len(text_predictions) / sampling_params.n:.2f} tokens")
        print(f"Average tokens per second: {total_tokens / inference_time:.2f} tokens/second")

        predictions = dict()
        for task_id, output, params in zip(task_ids, text_predictions, data_augmentation_params):
            predictions[task_id] = {
                'text_predictions': [output.text for output in output.outputs],
                'data_augmentation_params': params,
            }

        output_filepath = f'/mnt/hdd0/Kaggle/arc25/predictions/{experiment_name}/{dataset}_{sampling_params.n}preds_{get_timestamp()}_predictions.json'
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to {output_filepath}")


@log_execution_time
def load_vllm_model_and_tokenizer(model_path: str, use_4bit_quantization: bool=False, tensor_parallel_size: int=1,
               max_model_len: Optional[int]=None, enable_lora: bool=False, max_lora_rank: int=16):
    logger.info(f"Loading model from {model_path}")
    additional_kwargs = {}
    if max_model_len is not None:
        additional_kwargs['max_model_len'] = max_model_len
    if enable_lora:
        additional_kwargs['max_lora_rank'] = max_lora_rank
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.92,  # Use less GPU memory
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



if __name__ == '__main__':
    main()
