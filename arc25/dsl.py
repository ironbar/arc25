"""
ARC25 DSL

This module contains the DSL (Domain Specific Language) for the ARC25 challenge.
All the functions should have unit tests, that way I could refactor the code being sure that the code is working.

From time to time I should review the DSL to be sure that it is consistent and there is not any redundancy.

We can use [@singledispatch](https://docs.python.org/3/library/functools.html#functools.singledispatch)
to implement polymorphism in python.

## References

- [ARC24 DSL](https://github.com/ironbar/omni-arc/blob/main/omniarc/dsl.py)
- [DSL from BARC](https://github.com/xu3kev/BARC/blob/master/seeds/common.py)

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
from collections import deque
from dataclasses import dataclass
import numpy as np
import skimage
from scipy import stats
import scipy.ndimage

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
    return Img(np.rot90(img, k=n_rot90))


def flip(img: Img, axis: int) -> Img:
    return Img(np.flip(img, axis=axis))

#############################
# Math
#############################

def mode(x):
    return stats.mode(x, axis=None).mode

#############################
# Objects
#############################

@dataclass
class BoundingBox:
    min_row: int
    min_col: int
    max_row: int
    max_col: int

    def __repr__(self):
        return f"BoundingBox(min_row={self.min_row}, min_col={self.min_col}, max_row={self.max_row}, max_col={self.max_col})"

    @property
    def height(self):
        return int(self.max_row - self.min_row + 1)

    @property
    def width(self):
        return int(self.max_col - self.min_col + 1)

    @property
    def area(self):
        return int(self.height * self.width)

    @property
    def center(self):
        return np.array([(self.min_row + self.max_row) // 2, (self.min_col + self.max_col) // 2], dtype=int)

    def offset(self, offset):
        if isinstance(offset, int):
            offset = (offset, offset)
        return BoundingBox(self.min_row - offset[0], self.min_col - offset[1], self.max_row + offset[0], self.max_col + offset[1])

    def move(self, movement):
        return BoundingBox(self.min_row + movement[0], self.min_col + movement[1], self.max_row + movement[0], self.max_col + movement[1])

    @property
    def shape(self):
        return np.array([self.height, self.width], dtype=int)

    def __str__(self):
        return f"BoundingBox({self.min_row}, {self.min_col}, {self.max_row}, {self.max_col})"


class Object:
    def __init__(self, pixel_locations: list[tuple[int, int]], pixel_colors: list[int]):
        self.pixel_locations = np.array(pixel_locations, dtype=int)
        self.pixel_colors = pixel_colors
        self.area = len(pixel_locations)
        self.bounding_box = self._compute_bounding_box()

        self.colors = set(pixel_colors)
        if len(self.colors) == 1:
            self.color = self.pixel_colors[0]
        else:
            self.color = None

    def _compute_bounding_box(self):
        # Compute the bounding box: min_row, min_col, max_row, max_col
        min_row = min([x[0] for x in self.pixel_locations])
        max_row = max([x[0] for x in self.pixel_locations])
        min_col = min([x[1] for x in self.pixel_locations])
        max_col = max([x[1] for x in self.pixel_locations])
        return BoundingBox(min_row, min_col, max_row, max_col)

    @property
    def height(self):
        return self.bounding_box.height

    @property
    def width(self):
        return self.bounding_box.width

    @property
    def center(self):
        return np.mean(self.pixel_locations, axis=0).astype(int)

    @property
    def is_line(self):
        return self.is_vertical_line or self.is_horizontal_line

    @property
    def is_vertical_line(self):
        return self.width == 1 and self.height > 1

    @property
    def is_horizontal_line(self):
        return self.height == 1 and self.width > 1

    @property
    def is_point(self):
        return self.area == 1

    @property
    def is_square(self):
        return self.bounding_box.height == self.bounding_box.width and self.is_rectangle
    
    @property
    def is_rectangle(self):
        if self.is_point or self.is_line:
            return False
        is_filled_rectangle = self.area == self.bounding_box.height * self.bounding_box.width
        is_empty_rectangle = self.area == (self.bounding_box.height * 2 + self.bounding_box.width * 2 - 4)
        is_empty_rectangle = is_empty_rectangle and self.bounding_box.height >= 3 and self.bounding_box.width >= 3
        return is_filled_rectangle or is_empty_rectangle

    def is_in_img(self, img):
        return all(0 <= r < img.shape[0] and 0 <= c < img.shape[1] for r, c in self.pixel_locations)

    def is_inside(self, location):
        # TODO: move this functionality to the function below
        for pixel_location in self.pixel_locations:
            if pixel_location[0] == location[0] and pixel_location[1] == location[1]:
                return True
        return False

    def copy(self):
        return Object(self.pixel_locations.tolist(), self.pixel_colors.copy())
    
    # TODO: does this function belongs to the Object class?
    def move(self, movement):
        self.pixel_locations += np.array(movement)
        self.bounding_box = self._compute_bounding_box()

    # TODO: does this function belongs to the Object class?
    def change_color(self, color):
        self.color = color
        self.pixel_colors = [color] * self.area
        self.colors = set(self.pixel_colors)

    def count_holes(self):
        # Create a binary image of the object within its bounding box
        height = self.bounding_box.height
        width = self.bounding_box.width

        # Initialize an array of zeros (background)
        object_array = np.zeros((height, width), dtype=int)

        # Map the object's pixel locations to the array
        for r, c in self.pixel_locations:
            adjusted_r = r - self.bounding_box.min_row
            adjusted_c = c - self.bounding_box.min_col
            object_array[adjusted_r, adjusted_c] = 1  # Object pixel

        # Invert the object array: object pixels become 0, background becomes 1
        background_array = 1 - object_array

        # Create a visited mask for flood fill
        visited = np.zeros_like(background_array, dtype=bool)

        # Use a queue for BFS flood fill
        queue = deque()

        # Enqueue all border pixels that are background (external background)
        for i in range(height):
            for j in [0, width - 1]:
                if background_array[i, j] == 1 and not visited[i, j]:
                    queue.append((i, j))
                    visited[i, j] = True
        for j in range(width):
            for i in [0, height - 1]:
                if background_array[i, j] == 1 and not visited[i, j]:
                    queue.append((i, j))
                    visited[i, j] = True

        # Perform BFS to mark external background
        while queue:
            i, j = queue.popleft()
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < height and 0 <= nj < width:
                    if background_array[ni, nj] == 1 and not visited[ni, nj]:
                        visited[ni, nj] = True
                        queue.append((ni, nj))

        # The unvisited background pixels are holes
        holes_mask = (background_array == 1) & (~visited)

        # Label connected components in the holes_mask
        labeled_array, num_features = scipy.ndimage.label(holes_mask)

        return num_features


def detect_objects(image: Img, background_color: int = 0, 
                   connectivity: int = 4, monochrome: bool = True) -> list[Object]:
    """
    Detect objects in an ARC image using depth-first search (DFS).

    Parameters
    ----------
    image : Img
        The input image as a numpy array.
    background_color : int
        The color of the background pixels to be ignored. Defaults to 0.
    connectivity : int
        The connectivity of the objects to be filled. 4 for 4-connectivity, 8 for 8-connectivity.
    monochromatic : bool
        If True, only monochromatic objects (objects with the same color) are detected.
        If False, an object can have multiple colors.
    """
    # Directions for moving: (row_change, col_change) including diagonals
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError(f'Connectivity should be 4 or 8, got {connectivity}')

    def in_bounds(r, c):
        return 0 <= r < len(image) and 0 <= c < len(image[0])

    def dfs(r, c, visited, pixel_locations):
        visited[r][c] = True
        pixel_locations.append((r, c))
        current_color = image[r][c]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and not visited[nr][nc]:
                neighbor_color = image[nr][nc]
                if neighbor_color != background_color:
                    if monochrome:
                        if neighbor_color == current_color:
                            dfs(nr, nc, visited, pixel_locations)
                    else:
                        dfs(nr, nc, visited, pixel_locations)

    # Initialize variables
    visited = [[False] * len(image[0]) for _ in range(len(image))]
    objects = []

    # Loop through each pixel in the image
    for r in range(len(image)):
        for c in range(len(image[0])):
            if not visited[r][c] and image[r][c] != background_color:
                # If we find an unvisited pixel that is not the background color, start DFS
                pixel_locations = []
                dfs(r, c, visited, pixel_locations)

                # Create an Object and add it to the objects list
                pixel_colors = [image[r][c] for r, c in pixel_locations]
                obj = Object(pixel_locations, pixel_colors)
                objects.append(obj)

    return objects
