import tyro
import os
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

from arc25.logging import logging, configure_logging

logger = logging.getLogger(__name__)


def merge_lora(base_model_path: str, lora_path: str, output_path: str):
    """
    Merges a base model and a lora adapter into a single model.

    Args:
        base_model_path (str): Path to the folder with the base model.
        lora_path (str): Path to the folder with the lora adapter.
        output_path (str): Path to the folder where the merged model will be saved.
    """
    if is_lora_path(lora_path):
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if 'llama' in base_model_path:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            base_model.resize_token_embeddings(len(tokenizer))


        model = PeftModel.from_pretrained(base_model, lora_path)
        merged_model = model.merge_and_unload()
        logger.info('Saving the merged model to the output path')
        merged_model.save_pretrained(output_path)
        for filepath in glob.glob(os.path.join(lora_path, '*.json')):
            dst = os.path.join(output_path, os.path.basename(filepath))
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)
    else:
        logger.warning('The provided lora_path does not contain a lora adapter model, it is a full model')
        os.makedirs(output_path, exist_ok=True)
        for filepath in glob.glob(os.path.join(lora_path, '*')):
            dst = os.path.join(output_path, os.path.basename(filepath))
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)

    for filepath in glob.glob(os.path.join(base_model_path, '*')):
        dst = os.path.join(output_path, os.path.basename(filepath))
        if not os.path.exists(dst):
            logger.info(f'Copying {filepath}...')
            shutil.copy(filepath, dst)
    logger.info('Done!')


def is_lora_path(lora_path):
    return os.path.exists(os.path.join(lora_path, 'adapter_model.safetensors'))


if __name__ == '__main__':
    configure_logging()
    tyro.cli(merge_lora)
