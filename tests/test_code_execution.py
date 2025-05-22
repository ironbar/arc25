import pytest

from arc25.code_execution import (
    validate_code,
    check_code_is_safe,
    check_code_is_deterministic,
    InvalidCode,
    NonDeterministicCode,
    UnsafeCode,)
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
])
def test_validate_code_returns_validated_code(inputs, input_code, output_code):
    validated_code = validate_code(input_code, inputs)
    assert validated_code == output_code


@pytest.mark.parametrize("inputs, input_code", [
    # test that a line with the same color as background is removed
    ([create_img((3, 3), color=0)],
"""def task(img):
    draw_vertical_line(img, x=0, color=0)
    return img"""),
])
def test_validate_code_raises_exception_if_code_is_not_valid(inputs, input_code):
    with pytest.raises(InvalidCode):
        validate_code(input_code, inputs)


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
