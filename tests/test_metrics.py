import pytest
import numpy as np

from arc25.metrics import pixel_similarity_score

@pytest.mark.parametrize("ground_truth, prediction, expected_score", [
    # same size
    (np.ones((2, 2), dtype=int), np.ones((2, 2), dtype=int), 1.0),
    (np.ones((2, 2), dtype=int), np.zeros((2, 2), dtype=int), 0.0),
    # different sizes
    (np.ones((1, 1), dtype=int), np.ones((2, 2), dtype=int), 0.25),
    (np.ones((2, 2), dtype=int), np.ones((1, 1), dtype=int), 0.25),
    (np.ones((2, 1), dtype=int), np.ones((2, 2), dtype=int), 0.5),
    (np.ones((2, 2), dtype=int), np.ones((2, 1), dtype=int), 0.5),
    # different shapes but one side is bigger and the other is smaller
    (np.ones((2, 1), dtype=int), np.ones((1, 2), dtype=int), 0.25),
    (np.ones((1, 2), dtype=int), np.ones((2, 1), dtype=int), 0.25),
    (np.ones((2, 3), dtype=int), np.ones((3, 2), dtype=int), 4/9),
    # verify that works with 1d arrays
    (np.ones((1,), dtype=int), np.ones((2, 2), dtype=int), 0.25),
    (np.ones((2, 2), dtype=int), np.ones((1, ), dtype=int), 0.25),
    # edge cases
    (0, np.ones((2, 2), dtype=int), 0.0),
    (1, np.ones((2, 2), dtype=int), 0.25),
    (None, np.ones((2, 2), dtype=int), 0.0),
])
def test_pixel_similarity_score(ground_truth, prediction, expected_score):
    """
    Test the pixel similarity score function with given ground truth, prediction, and expected score.
    """
    score = pixel_similarity_score(ground_truth, prediction)
    assert pytest.approx(score, rel=1e-5) == expected_score