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

create_img, draw_line, draw_rectangle, flood_fill
"""
import numpy as np
import skimage
from typing import Tuple, Union

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


def create_img(shape: Tuple[int], color: int = 0) -> Img:
    return Img(np.ones(shape, dtype=np.int8) * color)


def draw_line(img: Img, point1: Tuple[int], point2: Tuple[int], color: int = 1) -> Img:
    rr, cc = skimage.draw.line(*point1, *point2)
    rr, cc = _filter_points_outside_the_image(rr, cc, img)
    img[rr, cc] = color
    return img


def draw_rectangle(img: Img, point1: Tuple[int], point2: Tuple[int], color: int = 1) -> Img:
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


def flood_fill(img: Img, point: Tuple[int], color: int, connectivity: int) -> Img:
    """
    Fill the area of the image with the given color starting from the given point.

    Parameters
    ----------
    connectivity : int
        The connectivity of the area to be filled. 4 for 4-connectivity, 8 for 8-connectivity.
    """
    mask = skimage.segmentation.flood(img, seed_point=point, connectivity=connectivity//4)
    img[mask] = color
    return img
