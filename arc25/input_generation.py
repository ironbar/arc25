import random
import logging
import math
from functools import reduce

from arc25.dsl import *

logger = logging.getLogger(__name__)


MAX_SIDE = 30


def random_draw_line_parameters(img: Img):
    point1 = (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
    line_type = random.choice(["horizontal", "vertical", "diagonal_decreasing", "diagonal_increasing"])

    # avoid infinite loops for diagonal lines
    if line_type == 'diagonal_increasing' and point1 == (0, 0) or point1 == (img.shape[0] - 1, img.shape[1] - 1):
        line_type = random.choice(["horizontal", "vertical", "diagonal_decreasing"])
    elif line_type == 'diagonal_decreasing' and point1 == (0, img.shape[1] - 1) or point1 == (img.shape[0] - 1, 0):
        line_type = random.choice(["horizontal", "vertical", "diagonal_increasing"])

    logging.debug(f"line_type: {line_type}")
    if line_type == "horizontal":
        while True:
            point2 = (point1[0], random.randint(0, img.shape[1] - 1))
            if point2 != point1:
                break
    elif line_type == "vertical":
        while True:
            point2 = (random.randint(0, img.shape[0] - 1), point1[1])
            if point2 != point1:
                break
    elif line_type == "diagonal_decreasing":
        while True:
            offset = random.randint(-min(point1), min(img.shape[0] - point1[0] - 1, img.shape[1] - point1[1] -1))
            point2 = (point1[0] + offset, point1[1] + offset)
            if point2 != point1:
                break
    elif line_type == "diagonal_increasing":
        while True:
            offset = random.randint(-min(img.shape[0] - point1[0] - 1, point1[1]), 
                                    min(point1[0], img.shape[1] - point1[1] - 1))
            point2 = (point1[0] - offset, point1[1] - offset)
            if point2 != point1:
                break

    point1, point2 = sorted([point1, point2])
    color = random.randint(0, 9)
    return dict(point1=point1, point2=point2, color=color)


def random_draw_rectangle_parameters(img: Img):
    point1 = (random.randint(0, img.shape[0] - 2), random.randint(0, img.shape[1] - 2))
    point2 = (random.randint(point1[0] + 1, img.shape[0] - 1), random.randint(point1[1] + 1, img.shape[1] - 1))
    color = random.randint(0, 9)
    return dict(point1=point1, point2=point2, color=color)


def random_draw_horizontal_line_parameters(img: Img):
    y = random.randint(0, img.shape[0] - 1)
    color = random.randint(0, 9)
    return dict(y=y, color=color)


def random_draw_vertical_line_parameters(img: Img):
    x = random.randint(0, img.shape[1] - 1)
    color = random.randint(0, 9)
    return dict(x=x, color=color)


def random_draw_pixel_parameters(img: Img):
    point = (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
    color = random.randint(0, 9)
    return dict(point=point, color=color)


def random_upscale_parameters(inputs: list[Img], max_upscale: int = 5):
    max_scale = np.min([MAX_SIDE // img.shape for img in inputs], axis=0)
    max_scale = np.minimum(max_scale, (max_upscale, max_upscale))
    scale = (random.randint(1, max_scale[0]), random.randint(1, max_scale[1]))
    return dict(scale=scale)


def random_downscale_parameters(inputs: list[Img], max_downscale: int = 3):
    # TODO: make this more robust
    scale = []
    for axis in range(2):
        sizes = [img.shape[axis] for img in inputs]
        scale.append(random.choice(common_divisors(sizes)))
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
