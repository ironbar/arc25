"""
Data augmentation module for ARC tasks.
It assumes that the input grids are numpy arrays, and the tasks are dictionaries
with 'train' and 'test' partitions, each containing a list of samples with 'input'
and 'output' grids.
"""
import random
import numpy as np
from functools import singledispatch
from typing import Optional, Union


@singledispatch
def apply_data_augmentation(task, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> dict:
    raise TypeError(f"Unsupported: {type(task).__name__}")

@apply_data_augmentation.register
def _(task: dict, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> dict:
    augmented_task = {partition: [{key: apply_data_augmentation(grid, hflip, n_rot90, color_map) for key, grid in sample.items()} \
                 for sample in samples] for partition, samples in task.items()}
    return augmented_task

@apply_data_augmentation.register
def _(grid: np.ndarray, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    grid = geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = apply_colormap(grid, color_map)
    return np.array(grid)

@apply_data_augmentation.register
def _(grid: list, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    return apply_data_augmentation(np.array(grid), hflip, n_rot90, color_map)


def revert_data_augmentation(grid, hflip, n_rot90, color_map=None):
    grid = revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = revert_color_swap(grid, color_map)
    return grid


def geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    if hflip:
        grid = np.flip(grid, axis=1)
    grid = np.rot90(grid, k=n_rot90)
    return grid


def revert_geometric_augmentation(grid, hflip, n_rot90):
    grid = np.array(grid)
    grid = np.rot90(grid, k=-n_rot90)
    if hflip:
        grid = np.flip(grid, axis=1)
    return grid


def revert_color_swap(grid, color_map):
    reverse_color_map = {v: int(k) for k, v in color_map.items()}
    vectorized_mapping = np.vectorize(reverse_color_map.get)
    return vectorized_mapping(grid)


def apply_colormap(grid, color_map):
    vectorized_mapping = np.vectorize(color_map.get)
    return vectorized_mapping(grid)


def get_random_data_augmentation_params():
    params = get_random_geometric_augmentation_params()
    params['color_map'] = get_random_color_map()
    return params


def get_random_geometric_augmentation_params():
    return dict(hflip=random.choice([True, False]), n_rot90=random.choice([0, 1, 2, 3]))


def get_random_color_map(change_background_probability=0.1):
    colors = list(range(10))
    if random.random() < change_background_probability:
        new_colors = list(range(10))
        random.shuffle(new_colors)
    else:
        new_colors = list(range(1, 10))
        random.shuffle(new_colors)
        new_colors = [0] + new_colors

    color_map = {x: y for x, y in zip(colors, new_colors)}
    return color_map