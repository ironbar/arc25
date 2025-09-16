import numpy as np
from typing import List

def validate_outputs(outputs: List[np.ndarray]) -> List[np.ndarray]:
    """
    Validate a list of output grids. Each output should be a 2D numpy array
    with integer values in range [0, 9], shape not larger than 30x30, and not empty.
    Raises ValueError if any output is invalid.
    Returns the list of validated outputs (as numpy arrays).
    """
    if not outputs:
        raise ValueError("Outputs list is empty")
    return [_validate_output(output) for output in outputs]


def _validate_output(output: np.ndarray) -> np.ndarray:
    if output is None:
        raise ValueError("Output is None")
    output = np.array(output, dtype=int) # otherwise I see weird outputs that mix list and numpy arrays
    if output.ndim != 2:
        raise ValueError(f"Output is not a 2D array. Output shape: {output.shape}")
    if max(output.shape) > 35:
        raise ValueError(f"Output is too large, the maximum allowed shape is 30x30. Output shape: {output.shape}")
    if min(output.shape) == 0:
        raise ValueError(f"Output has zero dimension, it is empty. Output shape: {output.shape}")
    if np.max(output) > 9 or np.min(output) < 0:
        raise ValueError(f"Output contains invalid values, expected values in range [0, 9]. Output max: {np.max(output)}, min: {np.min(output)}")
    # if not np.issubdtype(output.dtype, np.integer):
    #     raise ValueError(f"Output contains non-integer values, expected integer values. Output dtype: {output.dtype}")
    return output
