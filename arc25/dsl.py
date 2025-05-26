"""
ARC25 DSL

This module contains the DSL (Domain Specific Language) for the ARC25 challenge.
All the functions should have unit tests, that way I could refactor the code being sure that the code is working.

From time to time I should review the DSL to be sure that it is consistent and there is not any redundancy.

We can use [@singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch)
to implement polymorphism in python.

## Objects

This are the objects that can be used in the DSL.

- img: np.ndarray
- object:
- bounding_box: np.ndarray
- point: np.ndarray
- number: int

## Drawing functions

create_img, draw_line, draw_rectangle, flood_fill, draw_horizontal_line, draw_vertical_line, draw_pixel

TODO:
- Need a way to compare shapes of objects, even if they are upscaled or downscaled.
"""
import numpy as np
from typing import Union
import skimage
from scipy import stats

#############################
# Objects
#############################

class Img(np.ndarray):
    """
    A class that represents an image as a numpy array,
    and has a shape property that returns a numpy array.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    @property
    def shape(self):
        return np.array(super().shape)

    @shape.setter
    def shape(self, value):
        super(Img, self.__class__).shape.fset(self, tuple(value))

    def __repr__(self):
        return '\n'.join(''.join(str(int(v)) for v in row) for row in self)

    def __str__(self):
        return self.__repr__()

#############################
# Drawing functions
#############################

def create_img(shape: tuple[int, int], color: int = 0) -> Img:
    return Img(np.ones(shape, dtype=np.int8) * color)


def draw_line(img: Img, point1: tuple[int, int], point2: tuple[int, int], color: int) -> Img:
    rr, cc = skimage.draw.line(*point1, *point2)
    rr, cc = _filter_points_outside_the_image(rr, cc, img)
    img[rr, cc] = color
    return img


def draw_rectangle(img: Img, point1: tuple[int, int], point2: tuple[int, int], color: int) -> Img:
    rr, cc = skimage.draw.rectangle(point1, point2)
    rr, cc = _filter_points_outside_the_image(rr, cc, img)
    img[rr, cc] = color
    return img


def _filter_points_outside_the_image(rows, cols, img):
    valid_points = np.logical_and(np.logical_and(rows >= 0, rows < img.shape[0]),
                                  np.logical_and(cols >= 0, cols < img.shape[1]))
    rows = rows[valid_points]
    cols = cols[valid_points]
    return rows, cols


def flood_fill(img: Img, point: tuple[int, int], color: int, connectivity: int) -> Img:
    """
    Fill the area of the image with the given color starting from the given point.

    Parameters
    ----------
    connectivity : int
        The connectivity of the area to be filled. 4 for 4-connectivity, 8 for 8-connectivity.
    """
    # TODO: do I really need this function? I believe I could do the same with object detection and changing the color
    mask = skimage.segmentation.flood(img, seed_point=point, connectivity=connectivity//4)
    img[mask] = color
    return img


def draw_horizontal_line(img: Img, y: int, color: int) -> Img:
    img[y, :] = color
    return img


def draw_vertical_line(img: Img, x: int, color: int) -> Img:
    img[:, x] = color
    return img


def draw_pixel(img: Img, point: tuple[int, int], color: int) -> Img:
    if 0 <= point[0] < img.shape[0] and 0 <= point[1] < img.shape[1]:
        img[point[0], point[1]] = color
    return img

#############################
# Geometric transformations
#############################

def upscale(img: Img, scale: tuple[int, int]) -> Img:
    img = np.repeat(img, scale[0], axis=0)
    img = np.repeat(img, scale[1], axis=1)
    return Img(img)


def downscale(img: Img, scale: tuple[int, int]) -> Img:
    output = np.zeros((img.shape[0] // scale[0], img.shape[1] // scale[1]), dtype=img.dtype)
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            # TODO: maybe allow for other aggregation functions
            mode_result = mode(img[r*scale[0]:(r+1)*scale[0], c*scale[1]:(c+1)*scale[1]])
            output[r, c] = mode_result
    return Img(output)


def pad(img: Img, width: int, color: int) -> Img:
    return Img(np.pad(img, width, mode='constant', constant_values=color))


def trim(img: Img, width: int) -> Img:
    return img[width:-width, width:-width]


def rotate_90(img: Img, n_rot90: int) -> Img:
    return np.rot90(img, k=n_rot90)


def flip(img: Img, axis: int) -> Img:
    return np.flip(img, axis=axis)

#############################
# Math
#############################

def mode(x):
    return stats.mode(x, axis=None).mode
