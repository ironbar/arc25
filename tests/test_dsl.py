from arc25.dsl import *
import pytest


@pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
@pytest.mark.parametrize("color", [0, 1])
def test_create_img(shape, color):
    img = create_img(shape, color)
    assert isinstance(img, Img)
    assert np.all(img.shape == shape)
    assert np.all(img == color)


@pytest.mark.parametrize("input_img, point1, point2, color, output_img", [
    [create_img((3, 3), 0), (0, 0), (2, 2), 1, Img([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]]),],
    [create_img((3, 3), 0), (0, 0), (0, 2), 1, Img([[1, 1, 1],
                                                    [0, 0, 0],
                                                    [0, 0, 0]]),],
    [create_img((3, 3), 0), (0, 0), (2, 0), 1, Img([[1, 0, 0],
                                                    [1, 0, 0],
                                                    [1, 0, 0]]),],
    [create_img((3, 3), 0), (0, 0), (2, 0), 2, Img([[2, 0, 0],
                                                    [2, 0, 0],
                                                    [2, 0, 0]]),],
])
def test_draw_line(input_img, point1, point2, color, output_img):
    img = draw_line(input_img, point1, point2, color)
    assert np.array_equal(img, output_img)


@pytest.mark.parametrize("input_img, point1, point2, color, output_img", [
    [create_img((3, 3), 0), (0, 0), (1, 1), 1, Img([[1, 1, 0],
                                                    [1, 1, 0],
                                                    [0, 0, 0]]),],
    [create_img((3, 3), 0), (1, 1), (2, 2), 2, Img([[0, 0, 0],
                                                    [0, 2, 2],
                                                    [0, 2, 2]]),],
])
def test_draw_rectangle(input_img, point1, point2, color, output_img):
    img = draw_rectangle(input_img, point1, point2, color)
    assert np.array_equal(img, output_img)


@pytest.mark.parametrize("point, color, connectivity, input_img, output_img", [
    [(0, 0), 2, 4, Img([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]]), 
                    Img([[2, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]]),],
    [(0, 0), 2, 8, Img([[0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 0]]), 
                    Img([[2, 1, 2],
                         [1, 2, 2],
                         [2, 2, 2]]),],
])
def test_flood_fill(point, color, connectivity, input_img, output_img):
    img = flood_fill(input_img, point, color, connectivity)
    assert np.array_equal(img, output_img)


@pytest.mark.parametrize("y, color, input_img, output_img", [
    [1, 2, create_img((3, 3), 0), Img([[0, 0, 0],
                                       [2, 2, 2],
                                       [0, 0, 0]]),],
    [2, 3, create_img((3, 3), 0), Img([[0, 0, 0],
                                       [0, 0, 0],
                                       [3, 3, 3]]),],
])
def test_draw_horizontal_line(y, color, input_img, output_img):
    img = draw_horizontal_line(input_img, y, color)
    assert np.array_equal(img, output_img)


@pytest.mark.parametrize("x, color, input_img, output_img", [
    [1, 2, create_img((3, 3), 0), Img([[0, 2, 0],
                                       [0, 2, 0],
                                       [0, 2, 0]]),],
    [2, 3, create_img((3, 3), 0), Img([[0, 0, 3],
                                       [0, 0, 3],
                                       [0, 0, 3]]),],
])
def test_draw_vertical_line(x, color, input_img, output_img):
    img = draw_vertical_line(input_img, x, color)
    assert np.array_equal(img, output_img)


@pytest.mark.parametrize("point, color, input_img, output_img", [
    [(0, 0), 1, create_img((3, 3), 0), Img([[1, 0, 0],
                                            [0, 0, 0],
                                            [0, 0, 0]]),],
    [(1, 1), 2, create_img((3, 3), 0), Img([[0, 0, 0],
                                            [0, 2, 0],
                                            [0, 0, 0]]),],
])
def test_draw_pixel(point, color, input_img, output_img):
    img = draw_pixel(input_img, point, color)
    assert np.array_equal(img, output_img)
