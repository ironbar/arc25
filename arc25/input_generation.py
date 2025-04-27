import random
import logging

from arc25.dsl import *

logger = logging.getLogger(__name__)


def random_draw_line_parameters(img: Img):
    point1 = (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
    line_type = random.choice(["horizontal", "vertical", "diagonal_decreasing", "diagonal_increasing"])

    # avoid infinite loops for diagonal lines
    if line_type == 'diagonal_increasing' and point1 == (0, 0) or point1 == (img.shape[0] - 1, img.shape[1] - 1):
        line_type = random.choice(["horizontal", "vertical", "diagonal_decreasing"])
    elif line_type == 'diagonal_decreasing' and point1 == (0, img.shape[1] - 1) or point1 == (img.shape[0] - 1, 0):
        line_type = random.choice(["horizontal", "vertical", "diagonal_increasing"])

    logging.debug(f"line_type: {line_type}")
    if line_type == "horizontal":
        while True:
            point2 = (point1[0], random.randint(0, img.shape[1] - 1))
            if point2 != point1:
                break
    elif line_type == "vertical":
        while True:
            point2 = (random.randint(0, img.shape[0] - 1), point1[1])
            if point2 != point1:
                break
    elif line_type == "diagonal_decreasing":
        while True:
            offset = random.randint(-min(point1), min(img.shape[0] - point1[0] - 1, img.shape[1] - point1[1] -1))
            point2 = (point1[0] + offset, point1[1] + offset)
            if point2 != point1:
                break
    elif line_type == "diagonal_increasing":
        while True:
            offset = random.randint(-min(img.shape[0] - point1[0] - 1, point1[1]), 
                                    min(point1[0], img.shape[1] - point1[1] - 1))
            point2 = (point1[0] - offset, point1[1] - offset)
            if point2 != point1:
                break

    color = random.randint(0, 9)
    return dict(point1=point1, point2=point2, color=color)


def random_draw_rectangle_parameters(img: Img):
    point1 = (random.randint(0, img.shape[0] - 2), random.randint(0, img.shape[1] - 2))
    point2 = (random.randint(point1[0] + 1, img.shape[0] - 1), random.randint(point1[1] + 1, img.shape[1] - 1))
    color = random.randint(0, 9)
    return dict(point1=point1, point2=point2, color=color)


def random_draw_horizontal_line_parameters(img: Img):
    y = random.randint(0, img.shape[0] - 1)
    color = random.randint(0, 9)
    return dict(y=y, color=color)


def random_draw_vertical_line_parameters(img: Img):
    x = random.randint(0, img.shape[1] - 1)
    color = random.randint(0, 9)
    return dict(x=x, color=color)


def random_draw_pixel_parameters(img: Img):
    point = (random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1))
    color = random.randint(0, 9)
    return dict(point=point, color=color)

