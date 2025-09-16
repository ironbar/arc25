"""
Data augmentation module for ARC tasks.
It assumes that the input grids are numpy arrays, and the tasks are dictionaries
with 'train' and 'test' partitions, each containing a list of samples with 'input'
and 'output' grids.
"""
import random
import numpy as np
from functools import singledispatch
from typing import Optional, Dict


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
    grid = _geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = _apply_colormap(grid, color_map)
    return grid

@apply_data_augmentation.register
def _(grid: list, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    if isinstance(grid[0], list):
        return apply_data_augmentation(np.array(grid), hflip, n_rot90, color_map)
    elif isinstance(grid[0], np.ndarray):
        return [apply_data_augmentation(g, hflip, n_rot90, color_map) for g in grid]
    else:
        raise TypeError(f"Unsupported list element type: {grid}")

@singledispatch
def revert_data_augmentation(grid, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    raise TypeError(f"Unsupported: {type(grid).__name__}")

@revert_data_augmentation.register
def _(grid: np.ndarray, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    grid = _revert_geometric_augmentation(grid, hflip, n_rot90)
    if color_map is not None:
        grid = _revert_colormap(grid, color_map)
    return grid

@revert_data_augmentation.register
def _(grid: list, hflip: bool, n_rot90: int, color_map: Optional[dict]) -> np.ndarray:
    return [revert_data_augmentation(g, hflip, n_rot90, color_map) for g in grid]


def get_random_data_augmentation_params():
    params = _get_random_geometric_augmentation_params()
    params['color_map'] = _get_random_color_map()
    return params


def _geometric_augmentation(grid: np.ndarray, hflip: bool, n_rot90: int) -> np.ndarray:
    if hflip:
        grid = np.flip(grid, axis=1)
    grid = np.rot90(grid, k=n_rot90)
    return grid


def _revert_geometric_augmentation(grid: np.ndarray, hflip: bool, n_rot90: int) -> np.ndarray:
    grid = np.rot90(grid, k=-n_rot90)
    if hflip:
        grid = np.flip(grid, axis=1)
    return grid


def _apply_colormap(grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    vectorized_mapping = np.vectorize(color_map.get)
    return vectorized_mapping(grid)


def _revert_colormap(grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    reverse_color_map = {v: int(k) for k, v in color_map.items()}
    return _apply_colormap(grid, reverse_color_map)


def _get_random_geometric_augmentation_params() -> dict:
    return dict(hflip=random.choice([True, False]), n_rot90=random.choice([0, 1, 2, 3]))


def _get_random_color_map(change_background_probability: float = 0.1) -> Dict[int, int]:
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