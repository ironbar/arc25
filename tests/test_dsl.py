from arc25.dsl import *
import pytest


@pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
@pytest.mark.parametrize("color", [0, 1])
def test_create_img(shape, color):
    img = create_img(shape, color)
    assert isinstance(img, Img)
    assert np.all(img.shape == shape)
    assert np.all(img == color)

