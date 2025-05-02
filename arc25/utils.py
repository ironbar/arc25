"""
Functions commonly used in the challenge
"""
import random
import time
import numpy as np


def get_timestamp():
    time_stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    return time_stamp


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)