"""
Functions commonly used in the challenge
"""
import os
import random
import time
import numpy as np
import pynvml
import logging
import json
import gzip
from typing import Optional

logger = logging.getLogger(__name__)


def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


def set_cuda_visible_devices_to_least_used_gpu_if_undefined():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        logger.info(f'CUDA_VISIBLE_DEVICES is already set to {os.environ["CUDA_VISIBLE_DEVICES"]}, not changing it.')
        return
    index = get_least_used_gpu_index()
    logger.info(f'Setting CUDA_VISIBLE_DEVICES to {index}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(index)


def get_least_used_gpu_index():
    # TODO: now uses the GPU memory as a criteria, expand to other metrics
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    min_utilization = float('inf')
    best_gpu_index = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = mem_info.used / mem_info.total

        logger.info(f"GPU {i}: {utilization:.2%} used, {mem_info.used / (1024 ** 3):.2f} GB used, ")
        if utilization < min_utilization:
            min_utilization = utilization
            best_gpu_index = i

    pynvml.nvmlShutdown()
    logger.info(f"Least used GPU: {best_gpu_index} with {min_utilization:.2%} utilization")
    return best_gpu_index


def load_json(filepath):
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}')
    return data


def write_json(data, filepath):
    if filepath.endswith('.json.gz'):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    elif filepath.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f'Unsupported file extension: {filepath}')


def load_arc_dataset_with_solutions(filepath, convert_to_numpy=True):
    dataset = load_json(filepath)
    solutions_filepath = filepath.replace('challenges.json', 'solutions.json')
    assert solutions_filepath != filepath
    if os.path.exists(solutions_filepath):
        solutions = load_json(solutions_filepath)
        for sample_id, task in dataset.items():
            for idx, sample in enumerate(task['test']):
                sample['output'] = solutions[sample_id][idx]
        _verify_that_all_dataset_samples_have_output(dataset)
    else:
        logger.warning(f'Solutions file not found: {solutions_filepath}, loading dataset without solutions')
    if convert_to_numpy:
        for task_id, task in dataset.items():
            dataset[task_id] = {partition: [{key: np.array(value) for key, value in sample.items()} for sample in samples] for partition, samples in task.items()}
    return dataset


def _verify_that_all_dataset_samples_have_output(data):
    for task in data.values():
        if isinstance(task, dict):
            _verify_that_all_task_samples_have_outputs(task)
        elif isinstance(task, list):
            for subtask in task:
                _verify_that_all_task_samples_have_outputs(subtask)


def _verify_that_all_task_samples_have_outputs(task):
    for partition, samples in task.items():
        if partition not in ['train', 'test']:
            continue
        for sample in samples:
            if 'output' not in sample:
                raise ValueError('Not all samples have output')


def create_dataset_partitions(dataset_path, output_dir='dataset', n_partitions=4, max_tasks_per_partition: Optional[int]=None):
    logger.info(f'Creating {n_partitions} partitions from {dataset_path} into {output_dir}')
    dataset = load_arc_dataset_with_solutions(dataset_path, convert_to_numpy=False)
    task_ids = list(dataset.keys())
    task_ids = sorted(task_ids, key=lambda x: len(str(dataset[x])), reverse=True)
    partitions = [[] for _ in range(n_partitions)]
    partition_length = [0, 0, 0, 0]
    for task_id in task_ids:
        chosen_partition = np.argmin(partition_length)
        partitions[chosen_partition].append(task_id)
        partition_length[chosen_partition] += len(str(dataset[task_id]))
    logger.info(f'Partitions length: {partition_length}')
    logger.info(f'Number of tasks per partition: {[len(partition) for partition in partitions]}')
    assert len([task_id for partition in partitions for task_id in partition]) == len(dataset)
    os.makedirs(output_dir, exist_ok=True)
    for idx, partition in enumerate(partitions):
        if max_tasks_per_partition is not None:
            subset = {key: dataset[key] for key in partition[::-1][:max_tasks_per_partition]}
        else:
            subset = {key: dataset[key] for key in partition[::-1]}
        filepath = f'{output_dir}/partition{idx}_challenges.json'
        write_json(subset, filepath)
