import pytest

from arc25.data_augmentation import apply_data_augmentation

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
    hflip = True
    n_rot90 = 1
    color_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 4}

    apply_data_augmentation(task, hflip, n_rot90, color_map)
    apply_data_augmentation(task['train'][0]['input'], hflip, n_rot90, color_map)
