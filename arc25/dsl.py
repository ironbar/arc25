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

create grid, line, rectangle, fill
"""
import numpy as np


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


def create_img(shape, color):
    return Img(np.ones(shape, dtype=np.int8) * color)
