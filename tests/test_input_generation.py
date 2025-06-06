import pytest
from arc25.input_generation import generate_arc_image_with_random_objects
from arc25.dsl import detect_objects


@pytest.mark.parametrize("image_shape", [(10, 10), (15, 15)])
@pytest.mark.parametrize("n_objects", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("allowed_sizes", [[2], [2, 3]])
@pytest.mark.parametrize("allowed_colors", [None, [2]])
@pytest.mark.parametrize("monochrome", [True, False])
@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("background_color", [0, 1])
def test_generate_arc_image_with_random_objects(monochrome, connectivity, background_color,
                                                n_objects, image_shape, allowed_sizes, allowed_colors):
    img, placed = generate_arc_image_with_random_objects(
        monochrome=monochrome, connectivity=connectivity, background_color=background_color,
        n_objects=n_objects, image_shape=image_shape, allowed_sizes=allowed_sizes,
        allowed_colors=allowed_colors)
    print(image_shape, img.shape, placed)
    objects = detect_objects(img, monochrome=monochrome, connectivity=connectivity, background_color=background_color)
    assert len(objects) == n_objects, f"Expected {n_objects} objects, but found {len(objects)} on \n{img}"
