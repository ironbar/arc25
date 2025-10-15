import pytest
import numpy as np

from arc25.code_execution import (
    safe_code_execution,
    validate_code,
    check_code_is_safe,
    check_code_is_deterministic,
    InvalidCode,
    NonDeterministicCode,
    UnsafeCode,
    MemoryLimitExceeded)
from arc25.dsl import create_img

@pytest.mark.parametrize("inputs, input_code, output_code", [
    # test that a line with the same color as background is removed
    ([create_img((3, 3), color=0)],
"""def task(img):
    draw_vertical_line(img, x=0, color=0)
    draw_vertical_line(img, x=1, color=1)
    return img""",
"""def task(img):
    draw_vertical_line(img, x=1, color=1)
    return img"""),
    # test that a line that is ocluded by other line is removed
    ([create_img((3, 3), color=0)],
"""def task(img):
    draw_vertical_line(img, x=1, color=2)
    draw_vertical_line(img, x=1, color=1)
    return img""",
"""def task(img):
    draw_vertical_line(img, x=1, color=1)
    return img"""),
    # test that all lines are preserved if they change the image
    ([create_img((3, 3), color=0)],
"""def task(img):
    draw_vertical_line(img, x=0, color=2)
    draw_vertical_line(img, x=1, color=1)
    return img""",
"""def task(img):
    draw_vertical_line(img, x=0, color=2)
    draw_vertical_line(img, x=1, color=1)
    return img"""),
    # test that all lines are preserved if they change the image
    ([create_img((3, 3), color=0)],
"""def task(img):
    draw_vertical_line(img, x=0, color=2)
    return img""",
"""def task(img):
    draw_vertical_line(img, x=0, color=2)
    return img"""),
    # test that all lines are preserved if they change at least one image
    ([create_img((3, 3), color=0), create_img((3, 3), color=2)],
"""def task(img):
    draw_vertical_line(img, x=0, color=0)
    draw_vertical_line(img, x=1, color=1)
    return img""",
"""def task(img):
    draw_vertical_line(img, x=0, color=0)
    draw_vertical_line(img, x=1, color=1)
    return img"""),
    # test with output of different shape
    ([create_img((3, 3), color=0)],
"""def task(img):
    img = upscale(img, (2, 2))
    return img""",
"""def task(img):
    img = upscale(img, (2, 2))
    return img""",),
])
def test_validate_code_returns_validated_code(inputs, input_code, output_code):
    input_code = 'from arc25.dsl import *\n' + input_code
    output_code = 'from arc25.dsl import *\n' + output_code
    validated_code = validate_code(input_code, inputs)
    assert validated_code == output_code


@pytest.mark.parametrize("unsafe_code", [
    "import multiprocessing",
])
def test_check_code_is_safe_raises_exception_if_code_is_unsafe(unsafe_code):
    with pytest.raises(UnsafeCode):
        check_code_is_safe(unsafe_code)


@pytest.mark.parametrize("non_deterministic_code", [
    "import random",
    "np.random.randint(0, 10)",
])
def test_check_code_is_deterministic_raises_exception_if_code_is_non_deterministic(non_deterministic_code):
    with pytest.raises(NonDeterministicCode):
        check_code_is_deterministic(non_deterministic_code)


@pytest.mark.parametrize("execution_method", ['exec', 'subprocess'])
@pytest.mark.parametrize("code, expected_outputs", [
    ("""def task(img):
     return img * 2""",
        [np.array([[2, 0], [0, 2]])]),
    ("""def task(img):
     return img + 1""",
        [np.array([[2, 1], [1, 2]])]),
])
def test_safe_code_execution_returns_expected_output(code, expected_outputs, execution_method):
    inputs = [np.eye(2)]
    outputs = safe_code_execution(code, inputs, execution_method=execution_method)
    assert len(outputs) == len(expected_outputs)
    for output, expected_output in zip(outputs, expected_outputs):
        assert output.shape == expected_output.shape
        assert np.all(output == expected_output)


@pytest.mark.parametrize("execution_method", ['exec', 'subprocess'])
@pytest.mark.parametrize("mb", [2000, 3000, 4000]) # weirdly does not work correctly with 1000
def test_safe_code_execution_does_not_raise_memory_error(mb, execution_method):
    code = """
import numpy as np

def transform(input_grid):
    a = np.arange(N_MB * 1024 * 128)
    return input_grid.copy()
"""
    inputs = [np.zeros((10, 10), dtype=int)]
    code_mb = code.replace('N_MB', str(mb))
    safe_code_execution(code_mb, inputs, timeout_duration=10, func_name='transform', memory_limit_mb=mb*2, execution_method=execution_method)


@pytest.mark.parametrize("execution_method", ['exec', 'subprocess'])
@pytest.mark.parametrize("mb", [2000, 3000, 4000])
def test_safe_code_execution_raises_memory_error(mb, execution_method):
    code = """
import numpy as np

def transform(input_grid):
    a = np.arange(N_MB * 1024 * 128)
    return input_grid.copy()
"""
    inputs = [np.zeros((10, 10), dtype=int)]
    code_mb = code.replace('N_MB', str(mb))
    if execution_method == 'subprocess':
        with pytest.raises(RuntimeError):
            safe_code_execution(code_mb, inputs, timeout_duration=10, func_name='transform', memory_limit_mb=mb, execution_method=execution_method)
    elif execution_method == 'exec':
        with pytest.raises(MemoryLimitExceeded):
            safe_code_execution(code_mb, inputs, timeout_duration=10, func_name='transform', memory_limit_mb=mb, execution_method=execution_method)
