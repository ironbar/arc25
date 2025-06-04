from arc25.dsl import *
import pytest


@pytest.mark.parametrize("shape", [(1, 1), (2, 3)])
@pytest.mark.parametrize("color", [0, 1])
def test_create_img(shape, color):
    img = create_img(shape, color)
    assert isinstance(img, Img)
    assert np.all(img.shape == shape)
    assert np.all(img == color)
    print(img)


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


@pytest.mark.parametrize("img, background_color, connectivity, monochrome, n_objects", [
    # background color
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), 0, 4, False, 1),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), 1, 4, False, 2),
    # connectivity
    (Img([[1, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), 0, 4, False, 2),
    (Img([[1, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), 0, 8, False, 1),
    # monochrome
    (Img([[0, 0, 0],
          [0, 1, 2],
          [0, 1, 0]]), 0, 4, False, 1),
    (Img([[0, 0, 0],
          [0, 1, 2],
          [0, 1, 0]]), 0, 4, True, 2),
    # no objects
    (Img([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]), 0, 4, False, 0),
])
def test_detect_objects_returns_correct_number_of_objects(img, background_color, connectivity, monochrome, n_objects):
    objects = detect_objects(img, background_color, connectivity, monochrome)
    assert len(objects) == n_objects


@pytest.mark.parametrize("img, area, height, width", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), 3, 2, 2),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 1, 0]]), 2, 2, 1),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), 2, 1, 2),
])
def test_objects_have_correct_properties(img, area, height, width):
    object = detect_objects(img, background_color=0, connectivity=4, monochrome=False)[0]
    assert object.area == area
    assert object.height == height
    assert object.width == width

@pytest.mark.parametrize("img, is_line", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), True),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), True),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), True),
])
def test_objects_are_lines(img, is_line):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_line == is_line


@pytest.mark.parametrize("img, is_line", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), True),
])
def test_objects_are_vertical_lines(img, is_line):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_vertical_line == is_line


@pytest.mark.parametrize("img, is_line", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), True),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), True),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), False),
])
def test_objects_are_horizontal_lines(img, is_line):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_horizontal_line == is_line


@pytest.mark.parametrize("img, is_point", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), True),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), False),
    (Img([[0, 1, 0],
          [0, 0, 0],
          [0, 1, 0]]), True),
])
def test_objects_are_points(img, is_point):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_point == is_point


@pytest.mark.parametrize("img, is_square", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), False),
    (Img([[0, 1, 0],
          [0, 0, 0],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 1]]), True),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [1, 1, 1]]), False),
    (Img([[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]), True),
])
def test_objects_are_squares(img, is_square):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_square == is_square


@pytest.mark.parametrize("img, is_rectangle", [
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [0, 0, 0]]), False),
    (Img([[0, 1, 0],
          [0, 1, 0],
          [0, 1, 0]]), False),
    (Img([[0, 1, 0],
          [0, 0, 0],
          [0, 1, 0]]), False),
    (Img([[0, 0, 0],
          [0, 1, 1],
          [0, 1, 1]]), True),
    (Img([[0, 0, 0],
          [1, 1, 1],
          [1, 1, 1]]), True),
    (Img([[1, 1, 1],
          [1, 1, 1],
          [1, 1, 1]]), True),
])
def test_objects_are_rectangles(img, is_rectangle):
    objects = detect_objects(img, background_color=0, connectivity=4, monochrome=False)
    for obj in objects:
        assert obj.is_rectangle == is_rectangle


@pytest.mark.parametrize("img, center", [
      (Img([[0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]), (1, 1)),
      (Img([[0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]]), (1, 1)),
      (Img([[0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]]), (1, 1)),
      (Img([[0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]]), (1, 0)),
      (Img([[1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]]), (1, 0)),
])
def test_object_center(img, center):
    object = detect_objects(img, background_color=0, connectivity=4, monochrome=False)[0]
    assert np.array_equal(object.center, center)


@pytest.mark.parametrize("input_img, movement, output_img", [
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]),
     (1, 1),
     Img([[0, 0, 0],
          [0, 0, 0],
          [0, 0, 1]]),),
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]),
     (-1, -1),
     Img([[1, 0, 0],
          [0, 0, 0],
          [0, 0, 0]]),),
    (Img([[0, 0, 0],
          [1, 1, 0],
          [0, 0, 0]]),
     (-1, 0),
     Img([[1, 1, 0],
          [0, 0, 0],
          [0, 0, 0]]),),
])
def test_move_object(input_img, movement, output_img):
    object = detect_objects(input_img, background_color=0, connectivity=4, monochrome=False)[0]
    object.move(movement)
    moved_img = create_img(input_img.shape, 0)
    moved_img = draw_object(moved_img, object)
    assert np.array_equal(moved_img, output_img)


@pytest.mark.parametrize("input_img, color, output_img", [
    (Img([[0, 0, 0],
          [0, 1, 0],
          [0, 0, 0]]),
      2,
      Img([[0, 0, 0],
           [0, 2, 0],
           [0, 0, 0]]),)
])
def test_change_object_color(input_img, color, output_img):
    object = detect_objects(input_img, background_color=0, connectivity=4, monochrome=False)[0]
    object.change_color(color)
    new_img = create_img(input_img.shape, 0)
    new_img = draw_object(new_img, object)
    assert np.array_equal(new_img, output_img)


def test_object_copy():
    img = Img([[0, 0, 0],
               [0, 1, 0],
               [0, 0, 0]])
    object = detect_objects(img, background_color=0, connectivity=4, monochrome=False)[0]
    copied_object = object.copy()
    assert all(object.center == copied_object.center)
    copied_object.move((1, 1))
    assert all(object.center != copied_object.center)


@pytest.mark.parametrize("input_img, color_map, output_img", [
    (Img([[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8]]),
     {0:1, 1: 0},
     Img([[1, 0, 2],
          [3, 4, 5],
          [6, 7, 8]]),)
])
def test_apply_colormap(input_img, color_map, output_img):
    img = apply_colormap(input_img, color_map)
    assert np.array_equal(img, output_img)
