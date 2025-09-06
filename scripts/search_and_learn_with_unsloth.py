"""
Search and learn with unsloth
"""
import os
from arc25.utils import set_cuda_visible_devices_to_least_used_gpu_if_undefined
from arc25.logging import configure_logging, logging, log_execution_time

configure_logging()
set_cuda_visible_devices_to_least_used_gpu_if_undefined()

# Add VLLM specific environment variables to avoid common issues
os.environ['VLLM_USE_MODELSCOPE'] = 'False'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

from unsloth import FastLanguageModel

from dataclasses import dataclass
import sys
import random
import numpy as np
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import hashlib

import pandas as pd
from datasets import Dataset
import tyro
from vllm import  SamplingParams
from trl import SFTConfig, SFTTrainer
from tqdm.auto import tqdm
from accelerate import Accelerator

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions
from arc25.data_augmentation import apply_data_augmentation, revert_data_augmentation, get_random_data_augmentation_params
from arc25.code_execution import safe_code_execution
from arc25.prompting import create_prompt_from_task, parse_python_code_from_response, pretty_print_prompt
from arc25.metrics import pixel_similarity_score

from finetuning import get_data_collator # TODO: move to arc25 package

logger = logging.getLogger(__name__)

accelerator = Accelerator() # seems to need to do this if I want to logging


@dataclass
class Config:
    # base model
    model_path: str = "/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B"
    load_in_4bit: bool = False
    max_seq_length: int = 12000
    grid_encoder: str = 'ColorNameEncoder()'
    gpu_memory_utilization: float = 0.90
    # dataset
    dataset_path: str = "/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_training_challenges.json"
    # dataset_path: str = "/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json"
    max_epochs: int = 1
    use_data_augmentation: bool = True
    inference_batch_size: int = 4
    initial_predictions: int = 8
    predictions_per_epoch: int = 8
    training_batch_size: int = 1


def main():
    cfg = tyro.cli(Config, description="Search and learn with unsloth")
    logger.info(f'Running search and learn with config: {cfg}')


    assert cfg.predictions_per_epoch % cfg.inference_batch_size == 0


    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())[:2]
    print(f"Loaded {len(dataset)} tasks from {cfg.dataset_path}")


    llm, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_path, load_in_4bit=cfg.load_in_4bit, max_seq_length=cfg.max_seq_length,
        fast_inference=True, gpu_memory_utilization=cfg.gpu_memory_utilization)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)


    # on a first step, run inference on all tasks because it has more throughput
    prompts, data_augmentation_params, inference_task_ids = [], [], []
    for task_id in task_ids:
        task = dataset[task_id]
        for _ in range(cfg.initial_predictions // cfg.inference_batch_size):
            if cfg.use_data_augmentation:
                params = get_random_data_augmentation_params()
                task = apply_data_augmentation(task, **params)
            else:
                params = None
            data_augmentation_params.extend([params] * cfg.inference_batch_size)
            prompt = create_prompt_from_task(
                task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
            prompts.append(prompt)
            inference_task_ids.extend([task_id] * cfg.inference_batch_size)
    pretty_print_prompt(prompts[0])

    sampling_params = SamplingParams(n=cfg.inference_batch_size, temperature=1.0, top_p=0.95, max_tokens=2048)
    generations = llm.fast_generate(prompts, sampling_params)
    text_predictions = []
    for generation in generations:
        for output in generation.outputs:
            text_predictions.append(output.text)


    results = run_code_from_predictions(dataset, inference_task_ids, text_predictions, data_augmentation_params)


    print(aggregate_metrics(results))



    model = FastLanguageModel.get_peft_model(
        llm,
        r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = False, # True or "unsloth" for very long context
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    print(model.peft_config.keys())
    print(model.active_adapter)


    for task_id in tqdm(task_ids, desc="Tasks", unit="task"):
        print('\n'*2 + '='*80 + f'\nTask {task_id}\n' + '='*80)
        logger.info(f'Search and learn for task {task_id}')
        task = dataset[task_id]
        model.unload()
        model = FastLanguageModel.get_peft_model(
            llm,
            r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj'],
            lora_alpha = 64,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = False, # True or "unsloth" for very long context
            use_rslora = True,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        sampling_params = SamplingParams(n=cfg.inference_batch_size, temperature=1.0, top_p=0.95, max_tokens=2048) # TODO: move parameters to cfg
        task_results = results[task_id]
        for epoch in range(1, cfg.max_epochs + 1):
            logger.info(f'Prepare training data for epoch {epoch}')
            relabeled_tasks = create_hindsight_relabeled_tasks(task_results, task)
            training_prompts = create_training_prompts(relabeled_tasks, grid_encoder, tokenizer)
            train_dataset = Dataset.from_dict({'text': training_prompts})
            logger.info(f'Training on {len(train_dataset)} samples')
            trainer = SFTTrainer(
                model = model,
                tokenizer = tokenizer,
                train_dataset = train_dataset,
                dataset_text_field = "text",
                max_seq_length = 8192,
                packing = False, # Can make training 5x faster for short sequences.
                data_collator=get_data_collator(tokenizer),
                args = SFTConfig(
                    per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 1,
                    warmup_ratio=0.1,
                    num_train_epochs=1,
                    save_strategy='no',
                    learning_rate = 1e-5,
                    logging_steps = 1,
                    optim = "adamw_torch_fused",
                    weight_decay = 0.01,
                    lr_scheduler_type = 'constant_with_warmup',
                    # seed = 3407,
                    output_dir = "/mnt/hdd0/Kaggle/arc25/trainings/2025-09-04-debug-unsloth",
                    report_to = "none", # Use this for WandB etc
                ),
            )
            trainer_stats = trainer.train()
            model.save_lora("/mnt/hdd0/Kaggle/arc25/trainings/2025-09-05-debug-unsloth/lora")
            lora_request = model.load_lora("/mnt/hdd0/Kaggle/arc25/trainings/2025-09-05-debug-unsloth/lora")

            logger.info(f'Generating predictions for epoch {epoch}')
            prompts, data_augmentation_params = [], []
            for _ in range(cfg.predictions_per_epoch // cfg.inference_batch_size):
                if cfg.use_data_augmentation:
                    params = get_random_data_augmentation_params()
                    task = apply_data_augmentation(task, **params)
                else:
                    params = None
                data_augmentation_params.extend([params] * cfg.inference_batch_size)
                prompt = create_prompt_from_task(
                    task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
                prompts.append(prompt)
            generations = llm.fast_generate(prompts, sampling_params, lora_request=lora_request)

            text_predictions = []
            for generation in generations:
                for output in generation.outputs:
                    text_predictions.append(output.text)
            assert len(text_predictions) == cfg.predictions_per_epoch

            # run code and compute metrics
            task_results = run_code_from_predictions(dataset, [task_id]*len(text_predictions), text_predictions, data_augmentation_params)
            print(aggregate_metrics(task_results).head(1).round(3))
            task_results = task_results[task_id]
            results[task_id].extend(task_results)
            # TODO: stop criteria

    # TODO: select best predictions and prepare submission


    print(aggregate_metrics(results))


def curate_python_code(code):
    remove_line_keywords = ['import dsl', 'from dsl import ', 'print(', 'from common import *']
    code = '\n'.join(line for line in code.split('\n') if not any(keyword in line for keyword in remove_line_keywords))
    # code = 'from arc25.BARC_dsl import *\n' + code  # Ensure BARC_dsl is imported
    return code.strip()


def add_additional_imports(code):
    additional_imports = [
        'from typing import List, Tuple',
        'import numpy as np',
        'import numpy',
        'from arc25.BARC_dsl import *',
    ]
    imports = '\n'.join(additional_imports)
    return imports + '\n' + code if code else imports


def validate_outputs(outputs):
    if not outputs:
        raise ValueError("Outputs list is empty")
    return [_validate_output(output) for output in outputs]


def _validate_output(output):
    if output is None:
        raise ValueError("Output is None")
    output = np.array(output, dtype=int) # otherwise I see weird outputs that mix list and numpy arrays
    if output.ndim != 2:
        raise ValueError(f"Output is not a 2D array. Output shape: {output.shape}")
    if max(output.shape) > 35:
        raise ValueError(f"Output is too large, the maximum allowed shape is 30x30. Output shape: {output.shape}")
    if min(output.shape) == 0:
        raise ValueError(f"Output has zero dimension, it is empty. Output shape: {output.shape}")
    if np.max(output) > 9 or np.min(output) < 0:
        raise ValueError(f"Output contains invalid values, expected values in range [0, 9]. Output max: {np.max(output)}, min: {np.min(output)}")
    # if not np.issubdtype(output.dtype, np.integer):
    #     raise ValueError(f"Output contains non-integer values, expected integer values. Output dtype: {output.dtype}")
    return output


def run_code_from_predictions(dataset, task_ids, text_predictions, data_augmentation_params, n_jobs=-1):
    work = list(zip(text_predictions, [dataset[task_id] for task_id in task_ids], task_ids, data_augmentation_params))
    # sort the work by prediction index first and the task id second, I believe this will improve resource allocation
    # because some tasks are more resource intensive than others
    # work.sort(key=lambda x: (x[1], ))

    # with tqdm_joblib(tqdm(total=len(work), desc="Executing predictions", unit="pred")):
    with tqdm_joblib(total=len(work), desc="Executing code from predictions", unit="runs", smoothing=0):
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            prefer="processes",
            batch_size=1,
        )(delayed(_run_one)(*args) for args in work)
    grouped_results = {}
    for result in results:
        task_id = result.pop('task_id')
        if task_id not in grouped_results:
            grouped_results[task_id] = []
        grouped_results[task_id].append(result)
    return grouped_results


def _run_one(text_prediction, task, task_id, data_augmentation_params):
    code = parse_python_code_from_response(text_prediction)
    if not code:
        return dict(error_type="ParsingCodeFailed", error_message='', text_prediction=text_prediction,
                    task_id=task_id)
    try:
        input_grids = [sample['input'] for sample in task['train']] + [sample['input'] for sample in task['test']]
        if data_augmentation_params is not None:
            input_grids = apply_data_augmentation(input_grids, **data_augmentation_params)
        output_grids = safe_code_execution(
            add_additional_imports(curate_python_code(code)),
            input_grids,
            func_name="transform",
        )
        output_grids = validate_outputs(output_grids)
        if data_augmentation_params is not None:
            original_output_grids = revert_data_augmentation(output_grids, **data_augmentation_params)
        else:
            original_output_grids = output_grids
        result = dict(code=code, output_grids=output_grids,
                      input_grids=input_grids, text_prediction=text_prediction,
                      fingerprint=fingerprint(original_output_grids),
                      task_id=task_id)
        result.update(_compute_metrics(task, original_output_grids))
        return result
    except Exception as e:
        return dict(code=code, error_type=type(e).__name__, error_message=str(e), task_id=task_id)


def fingerprint(prediction):
    """
    Create a compact hash for a list of matrices.
    Includes shape & dtype to distinguish e.g. (2×2) from (4×1).
    """
    h = hashlib.sha256()
    for m in prediction:
        # incorporate shape and dtype in a reproducible way
        h.update(str(m.shape).encode())
        h.update(m.dtype.str.encode())
        # raw data bytes
        h.update(m.tobytes())
    return h.hexdigest()


def _compute_metrics(task, predicted_grids):
    metrics = {}
    for partition in ['train', 'test']:
        if not 'output' in task[partition][0]:
            continue # we won't have the output when making submissions
        gt_grids = [sample['output'] for sample in task[partition]]
        n_samples = len(gt_grids)
        partition_predicted_grids = predicted_grids[:n_samples] if partition == 'train' else predicted_grids[-n_samples:]
        pixel_scores = np.array([pixel_similarity_score(pred, gt) for pred, gt in zip(partition_predicted_grids, gt_grids)])
        metrics[f"{partition}_pixel_score"] = float(np.mean(pixel_scores))
        metrics[f'{partition}_correct_grids'] = float(np.mean(pixel_scores == 1))
        metrics[f'{partition}_is_correct'] = int(all(pixel_scores == 1))
    return metrics


def aggregate_metrics(results):
    df = pd.DataFrame()
    for task_id, task_results in results.items():
        n_preds = len(task_results)
        df.loc[task_id, 'n_preds'] = n_preds
        df.loc[task_id, 'valid code'] = (len([1 for result in task_results if 'code' in result]))/n_preds
        df.loc[task_id, 'valid outputs'] = (len([1 for result in task_results if 'error_type' not in result]))/n_preds
        df.loc[task_id, 'unique outputs'] = len(set(result['fingerprint'] for result in task_results if 'fingerprint' in result))/n_preds
        for partition in ['train', 'test']:
            df.loc[task_id, f'{partition}_pixel_score'] = np.mean([result.get(f'{partition}_pixel_score', 0) for result in task_results])
            df.loc[task_id, f'{partition}_correct_grids'] = np.mean([result.get(f'{partition}_correct_grids', 0) for result in task_results])
            df.loc[task_id, f'{partition}_pass_rate'] = sum(result.get(f'{partition}_is_correct', 0) for result in task_results)/n_preds
            df.loc[task_id, f'{partition}_is_correct'] = int(any(result.get(f'{partition}_is_correct', 0) for result in task_results))
    if 'test_is_correct' in df.columns:
        df['is_correct'] = df['train_is_correct'] * df['test_is_correct']
    df.loc['MEAN'] = df.mean(axis=0)
    return df.astype(float)


def error_analysis(results):
    errors_to_check = ['TimeoutException', 'NonDeterministicCode', 'UnsafeCode', 'ParsingCodeFailed']

    df = pd.DataFrame(columns=['n_preds', 'error_rate'] + errors_to_check)
    all_errors = []
    for task_id, task_results in results.items():
        task_errors = [result['error_type'] for result in task_results if 'error_type' in result]
        all_errors.extend(task_errors)
        df.loc[task_id, 'n_preds'] = len(task_results)
        df.loc[task_id, 'error_rate'] = len(task_errors) / len(task_results) if task_results else 0.0
        for error_type in errors_to_check:
            df.loc[task_id, error_type] = sum(1 for error in task_errors if error == error_type) / len(task_results) if task_results else 0.0
    df.loc['MEAN'] = df.mean(axis=0)

    error_counts = pd.Series(all_errors).value_counts()
    print("Most common errors:")
    display(error_counts.head(20))
    return df.astype(float)


def create_hindsight_relabeled_tasks(results, task):
    # TODO: strategies to avoid repetitions
    # sort the tasks, placing the best ones last
    sorted_results = sorted(results, key=lambda r: (r.get('train_correct_grids', -1), r.get('train_pixel_score', -1)), reverse=False)
    relabeled_tasks = []
    n_train = len(task['train'])
    for result in sorted_results:
        if 'output_grids' not in result:
            continue
        new_task = {
            'train': [{'input': input, 'output': output} for input, output in zip(result['input_grids'][:n_train], result['output_grids'][:n_train])],
            'test': [{'input': input, 'output': output} for input, output in zip(result['input_grids'][n_train:], result['output_grids'][n_train:])],
            'text_prediction': result['text_prediction'],
        }
        relabeled_tasks.append(new_task)
    return relabeled_tasks


def create_training_prompts(relabeled_tasks, grid_encoder, tokenizer):
    prompts = []
    for task in relabeled_tasks:
        prompt = create_prompt_from_task(
            task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
        prompt += task['text_prediction'] + tokenizer.eos_token
        prompts.append(prompt)
    return prompts


if __name__ == '__main__':
    main()
