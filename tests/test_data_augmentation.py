import pytest
import numpy as np

from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params, revert_data_augmentation

def test_apply_data_augmentation_is_polymorphic():
    task = {
        'train': [
            {'input': [[1, 2], [3, 4]], 'output': [[4, 3], [2, 1]]},
            {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
        ],
        'test': [
            {'input': [[2, 2], [2, 2]], 'output': [[2, 2], [2, 2]]}
        ]
    }
    apply_data_augmentation(task, **get_random_data_augmentation_params())
    apply_data_augmentation(task['train'][0]['input'], **get_random_data_augmentation_params())


def test_data_augmentation_does_not_modify_original():
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for _ in range(10):
        params = get_random_data_augmentation_params()
        augmented_grid = apply_data_augmentation(grid, **params)
        assert not np.array_equal(grid, augmented_grid), "Augmented grid should differ from original"
        assert np.array_equal(grid, np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])), "Original grid was modified"


def test_revert_data_augmentation():
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for _ in range(10):
        params = get_random_data_augmentation_params()
        augmented_grid = apply_data_augmentation(grid, **params)
        assert not np.array_equal(grid, augmented_grid), "Augmented grid should differ from original"
        reverted_grid = revert_data_augmentation(augmented_grid, **params)
        assert np.array_equal(grid, reverted_grid), "Reverted grid does not match original"
