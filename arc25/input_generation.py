import random
import logging
import math
from typing import Optional
from functools import reduce
import cv2

from arc25.dsl import *
from arc25.constants import MAX_SIDE

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

    point1, point2 = sorted([point1, point2])
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


"""
Random structure objects image
"""
class NoPositionAvailable(Exception):
    pass


def create_image_with_random_objects(shape, n_objects=None, color_range=None, background_ratio=None):
    n_objects = n_objects or random.randint(1, int(np.sqrt(np.prod(shape))))
    if color_range is None:
        monochrome = random.random() < 0.5
        if monochrome:
            color = random.randint(1, 9)
            color_range = [color, color]
        else:
            color_range = [1, 9]
    background_ratio = background_ratio or random.uniform(0.25, 0.75)

    img = create_img(shape)
    objects = []
    for idx in range(n_objects):
        try:
            background_to_fill = np.mean(img == 0) - background_ratio
            if idx < n_objects - 1:
                background_to_fill = random.uniform(0, background_to_fill)
            object_desired_area = max(int(np.prod(img.shape)*background_to_fill), 1)
            objects.append(add_random_object(img, color_range=color_range, area=object_desired_area))
        except NoPositionAvailable:
            pass
    return img


def add_random_object(grid, color_range, area):
    """
    To ensure connectivity the pixels of the object are created sequentially, always on the neighbor pixels of the last pixel
    """
    color = random.randint(*color_range)
    dilated_grid = cv2.dilate((grid == color).astype(np.uint8), np.ones((3, 3), np.uint8))
    
    position = get_initial_position(grid + dilated_grid)
    grid[position[0], position[1]] = color
    positions = [position]
    
    while len(positions) < area:
        candidate_positions = get_candidate_positions(position, grid + dilated_grid)
        if not candidate_positions:
            break
        position = random.choice(candidate_positions)
        grid[position[0], position[1]] = color
        positions.append(position)
    
    return dict(color=color, positions=positions)


def get_initial_position(grid):
    candidate_positions = np.where(grid == 0)
    if candidate_positions[0].size == 0:
        raise NoPositionAvailable
    chosen_idx = random.randint(0, len(candidate_positions[0]) - 1)
    return int(candidate_positions[0][chosen_idx]), int(candidate_positions[1][chosen_idx])
    
    
def get_candidate_positions(position, grid):
    # TODO: I might speed this by using arrays, selecting the candidate positions at once.
    candidate_positions = []
    for i in range(position[0] - 1, position[0] + 2):
        if i < 0 or i >= grid.shape[0]:
            continue
        for j in range(position[1] - 1, position[1] + 2):
            if j < 0 or j >= grid.shape[1]:
                continue
            if grid[i, j] == 0:
                candidate_positions.append((i, j))
    return candidate_positions


def generate_arc_image_with_random_objects(
    image_shape: tuple[int, int],
    allowed_sizes: list[int],
    n_objects: int,
    monochrome: bool = True,
    connectivity: int = 4,
    background_color: int = 0,
    allowed_colors: Optional[list[int]] = None,
    max_attempts: int = 1_000,
    random_shape_probability: float = 0.5,
    line_shape_probability: float = 0.5,
):
    """
    Random ARC-style image generator.

    An object of size *s* is produced in two ways:
      • with probability ``random_shape_probability`` a free-form random shape
      • otherwise a regular shape chosen at random from
        {vertical_line, horizontal_line, rectangle}.

    The remainder of the behaviour matches the original implementation.

    Parameters
    ----------
    image_shape      : (height, width) of the grid.
    allowed_sizes    : admissible object sizes (number of cells).
    n_objects        : number of connected components to place.
    monochrome
        True  → each object is a single colour; objects may touch,
                but any touching neighbours must be of *different* colours.
        False → each object may be internally multi-coloured; objects may
                NOT touch any other object.
    connectivity     : 4 (von Neumann) or 8 (Moore) adjacency.
    background_color : value used for empty cells (default 0).
    allowed_colors   : list of usable colours for objects; if None
                       it defaults to all digits 0-9 except background.
    max_attempts     : cap on placement attempts.
    random_shape_probability : chance of drawing a free-form shape (0–1).
    line_shape_probability : chance of drawing a line shape (0–1), when regular shape is chosen

    Returns
    -------
    Img              : generated image.
    int              : number of objects placed.
    """
    if not 0.0 <= random_shape_probability <= 1.0:
        raise ValueError("random_shape_probability must lie in [0, 1]")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if not 0 <= background_color <= 9:
        raise ValueError("background_color must be in 0–9")

    palette = (
        [c for c in range(10) if c != background_color]
        if allowed_colors is None
        else [c for c in allowed_colors if c != background_color]
    )
    if not palette:
        raise ValueError("allowed_colors empty after removing background")

    H, W = image_shape
    grid = np.full((H, W), background_color, dtype=int)

    offs4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    offs8 = offs4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    offs = offs4 if connectivity == 4 else offs8

    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    def free(cells):
        return all(grid[r, c] == background_color for r, c in cells)

    # ----- shape generators -------------------------------------------------
    def shape_random(size):
        free_cells = np.argwhere(grid == background_color)
        if free_cells.size == 0:
            return []
        shape = [tuple(random.choice(free_cells))]
        frontier = {shape[0]}
        while len(shape) < size:
            cand = set()
            for r0, c0 in frontier:
                for dr, dc in offs:
                    r1, c1 = r0 + dr, c0 + dc
                    if (
                        in_bounds(r1, c1)
                        and grid[r1, c1] == background_color
                        and (r1, c1) not in shape
                    ):
                        cand.add((r1, c1))
            if not cand:
                return []
            nxt = random.choice(list(cand))
            shape.append(nxt)
            frontier.add(nxt)
        return shape

    def shape_vertical_line(size):
        if size > H:
            return []
        for _ in range(10):
            col = random.randrange(W)
            row0 = random.randrange(H - size + 1)
            cells = [(row0 + i, col) for i in range(size)]
            if free(cells):
                return cells
        return []

    def shape_horizontal_line(size):
        if size > W:
            return []
        for _ in range(10):
            row = random.randrange(H)
            col0 = random.randrange(W - size + 1)
            cells = [(row, col0 + i) for i in range(size)]
            if free(cells):
                return cells
        return []

    def shape_rectangle(size):
        factors = [
            (h, size // h)
            for h in range(2, int(size**0.5) + 1)
            if size % h == 0
        ]
        random.shuffle(factors)
        for h, w in factors:
            if h > H or w > W:
                continue
            for _ in range(10):
                r0 = random.randrange(H - h + 1)
                c0 = random.randrange(W - w + 1)
                cells = [
                    (r0 + dr, c0 + dc) for dr in range(h) for dc in range(w)
                ]
                if free(cells):
                    return cells
        return []

    def make_shape(size):
        if random.random() < random_shape_probability:
            return shape_random(size)
        if random.random() < line_shape_probability:
            gen = random.choice([shape_vertical_line, shape_horizontal_line])
            return gen(size)
        else:
            return shape_rectangle(size)

    # ----- neighbour-colour helper ------------------------------------------
    def neighbour_colours(shape):
        s = set()
        for r, c in shape:
            for dr, dc in offs:
                r1, c1 = r + dr, c + dc
                if in_bounds(r1, c1):
                    col = grid[r1, c1]
                    if col != background_color:
                        s.add(col)
        return s

    placed, attempts = 0, 0
    while placed < n_objects and attempts < max_attempts:
        attempts += 1
        size = random.choice(allowed_sizes)
        shape = make_shape(size)
        if not shape:
            continue
        neigh = neighbour_colours(shape)
        if monochrome:
            free_cols = [c for c in palette if c not in neigh]
            if not free_cols:
                continue
            colour = random.choice(free_cols)
            for r, c in shape:
                grid[r, c] = colour
        else:
            if neigh:
                continue
            for r, c in shape:
                grid[r, c] = random.choice(palette)
        placed += 1

    return Img(grid), placed


def create_inputs_generate_arc_image_with_random_objects(
        min_inputs: int, max_inputs: int, min_side: int, max_side: int, allowed_sizes: list[int],
        min_objects: int, max_objects: int, n_allowed_colors: Optional[int] = None,
        random_shape_probability: float = 0.5,
        line_shape_probability: float = 0.5,
        **kwargs):
    n_inputs = random.randint(min_inputs, max_inputs)
    shapes = [np.random.randint(min_side, max_side + 1, 2) for _ in range(n_inputs)]
    metadata = dict(allowed_sizes=allowed_sizes,
                    connectivity=random.choice([4, 8]),
                    monochrome=random.choice([True, False]),
                    background_color=random.choice([0]*18 + list(range(1, 10))),
                    random_shape_probability=random_shape_probability,
                    line_shape_probability=line_shape_probability)
    if n_allowed_colors is not None:
        allowed_colors = random.sample([color for color in range(10) if color != metadata['background_color']], n_allowed_colors)
    else:
        allowed_colors = None
    inputs = [generate_arc_image_with_random_objects(
        shape, **metadata, n_objects=random.randint(min_objects, max_objects), 
        allowed_colors=allowed_colors)[0]
        for shape in shapes]
    return inputs, metadata
