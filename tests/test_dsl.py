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


def test_upscale_and_downscale():
    img = create_img((2, 2), 1)
    upscaled_img = upscale(img, (2, 2))
    assert np.array_equal(upscaled_img, create_img((4, 4), 1))

    downscaled_img = downscale(upscaled_img, (2, 2))
    assert np.array_equal(downscaled_img, img)


def test_pad_and_trim():
    img = create_img((2, 2), 1)
    padded_img = pad(img, 1, 0)
    assert np.array_equal(padded_img, Img([[0, 0, 0, 0],
                                            [0, 1, 1, 0],
                                            [0, 1, 1, 0],
                                            [0, 0, 0, 0]]))
    trimmed_img = trim(padded_img, 1)
    assert np.array_equal(trimmed_img, img)


def test_rotate():
    img = Img([[1, 2],
                [3, 4]])
    for n_rot90 in range(1, 5):
        rotated_img = rotate_90(img, n_rot90)
        if n_rot90 < 4:
            assert not np.array_equal(rotated_img, img)
        else:
            assert np.array_equal(rotated_img, img)


def test_flip():
    img = Img([[1, 2],
                [3, 4]])
    for axis in range(2):
        flipped_img = flip(img, axis)
        assert all(flipped_img.shape == img.shape)
        assert not np.array_equal(flipped_img, img)
        flipped_img_again = flip(flipped_img, axis)
        assert np.array_equal(flipped_img_again, img)
