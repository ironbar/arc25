"""
Functions commonly used in the challenge
"""
import random
import time
import numpy as np
import pynvml
import logging

logger = logging.getLogger(__name__)


def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)


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
