import random
import numpy as np
import math
from functools import reduce

from arc25.constants import MAX_SIDE
from arc25.dsl import Img


def random_upscale_parameters(inputs: list[Img], max_upscale: int = 5):
    max_scale = np.min([MAX_SIDE // img.shape for img in inputs], axis=0)
    max_scale = np.minimum(max_scale, (max_upscale, max_upscale))
    scale = (random.randint(1, max_scale[0]), random.randint(1, max_scale[1]))
    return dict(scale=scale)


def random_downscale_parameters(inputs: list[Img], max_tries: int = 5):
    for _ in range(max_tries):
        scale = []
        for axis in range(2):
            sizes = [img.shape[axis] for img in inputs]
            scale.append(random.choice(common_divisors(sizes)))
        if scale != [1, 1]:
            break
    return dict(scale=tuple(scale))


def common_divisors(numbers: list[int]) -> list[int]:
    """
    Given a list of positive integers (each ≤ 30), return a sorted list
    of all divisors that divide every number in the list.

    Example:
        >>> common_divisors([12, 18, 24])
        [1, 2, 3, 6]
    """
    if not numbers:
        return []

    # Compute GCD of the whole list
    gcd_all = reduce(math.gcd, numbers)

    # Find all divisors of gcd_all
    divs = []
    for i in range(1, gcd_all + 1):
        if gcd_all % i == 0:
            divs.append(i)

    return divs


def random_pad_parameters(inputs: list[Img], max_width: int = 5):
    max_possible_width = min(min(MAX_SIDE - img.shape) for img in inputs)//2
    if max_possible_width <= 0:
        raise ValueError("All input images are too large to pad.")
    width = random.randint(1, min(max_width, max_possible_width))
    color = random.randint(0, 9)
    return dict(width=width, color=color)


def random_trim_parameters(inputs: list[Img], max_width: int = 5):
    max_possible_width = min(min(img.shape) - 1 for img in inputs)//2
    width = random.randint(1, min(max_width, max_possible_width))
    return dict(width=width)


def random_rotate_90_parameters(*args, **kwargs):
    n_rot90 = random.randint(1, 3)  # 1 to 3 rotations of 90 degrees
    return dict(n_rot90=n_rot90)


def random_flip_parameters(*args, **kwargs):
    axis = random.choice([0, 1])  # 0 for vertical flip, 1 for horizontal flip
    return dict(axis=axis)
