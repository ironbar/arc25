"""
DSL Training tasks

All training tasks return: inputs, outputs and code

Each training task should teach a specific concept, and the name of the task should reflect that
"""
import random
import sys
import inspect
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from typing import Union
from arc25.dsl import *
from arc25.input_generation import *
from arc25.code_execution import safe_code_execution, validate_code, wrap_code_in_function, InvalidCode


logger = logging.getLogger(__name__)


Task = namedtuple("Task", ["inputs", "outputs", "code", 'name'])


class TrainingTask(ABC):
    def sample(self, n_tries=3):
        ret = self.create_inputs()
        if isinstance(ret, tuple):
            inputs = ret[0]
            metadata = ret[1:]
        else:
            inputs = ret
            metadata = None

        is_valid_code = False
        for _ in range(n_tries):
            try:
                # TODO: better handling, what happens if we reach n_tries?
                if metadata is not None:
                    code = self.create_code(inputs, *metadata)
                else:
                    code = self.create_code(inputs)
                code = validate_code(code, inputs)
                is_valid_code = True
                break
            except InvalidCode as e:
                logger.debug(f"{e}:\n{code}\nRetrying...")
            except Exception as e:
                logger.error(f"Unexpected error: {e}:\nRetrying...")
                logger.error(traceback.format_exc())
        if not is_valid_code:
            raise InvalidCode(f"Failed to create a valid code for task {self.__class__.__name__} after {n_tries} attempts.")
        outputs = safe_code_execution(code, inputs)
        return Task(inputs=inputs, outputs=outputs, code=code, name=self.__class__.__name__)

    @abstractmethod
    def create_inputs(self) -> Union[list[Img], tuple[list[Img], ...]]:
        pass

    @abstractmethod
    def create_code(self, inputs: list[Img], *metadata) -> str:
        pass


def training_tasks_generator():
    training_classes = _get_all_training_classes()
    training_tasks = [cls() for cls in training_classes]
    logger.info(f"Found {len(training_tasks)} training tasks: {[task.__class__.__name__ for task in training_tasks]}")
    while True:
        # TODO: in the future I would like to be able to give weighted probabilities to the tasks
        task = random.choice(training_tasks)
        yield task.sample()


def _get_all_training_classes():
    current_module = sys.modules[__name__]
    training_classes = [
        cls for name, cls in inspect.getmembers(current_module, inspect.isclass)
        if issubclass(cls, TrainingTask)
        and cls is not TrainingTask
        and cls.__module__ == __name__
    ]
    return sorted(training_classes, key=lambda x: x.__name__)


@dataclass
class DrawingTaskOnEmptyImg(TrainingTask):
    n_inputs: int = 1
    min_draws: int = 1
    max_draws: int = 5
    min_side: int = 3
    max_side: int = 10

    def create_inputs(self):
        shape = np.random.randint(self.min_side, self.max_side + 1, 2)
        colors = random.sample(range(10), self.n_inputs)
        return [create_img(shape, color=color) for color in colors]

    def create_code(self, inputs):
        n_draws = random.randint(self.min_draws, self.max_draws)
        parameter_functions = [
            random_draw_line_parameters,
            random_draw_rectangle_parameters,
            random_draw_horizontal_line_parameters,
            random_draw_vertical_line_parameters,
            random_draw_pixel_parameters
        ]
        code = ''
        for _ in range(n_draws):
            parameter_function = random.choice(parameter_functions)
            parameters = parameter_function(inputs[0])
            function_name = parameter_function.__name__.replace("random_", "").replace("_parameters", "")
            code += f"{function_name}(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code = wrap_code_in_function(code)
        return code


@dataclass
class DrawingTaskOnEmptyImgs(DrawingTaskOnEmptyImg):
    n_inputs = 2


@dataclass
class DrawingTaskOnRandomImgs(DrawingTaskOnEmptyImg):
    n_inputs = 3

    def create_inputs(self):
        shape = np.random.randint(self.min_side, self.max_side + 1, 2)
        return [Img(np.random.randint(0, 10, size=shape)) for _ in range(self.n_inputs)]


@dataclass
class DrawingTaskOnObjectImgs(DrawingTaskOnEmptyImg):
    n_inputs = 3

    def create_inputs(self):
        shape = np.random.randint(self.min_side, self.max_side + 1, 2)
        return [create_image_with_random_objects(shape) for _ in range(self.n_inputs)]


@dataclass
class ShapeDependentDrawings(DrawingTaskOnEmptyImg):
    min_inputs: int = 2
    max_inputs: int = 5
    min_side: int = 3
    max_side: int = 10
    min_draws: int = 1
    max_draws: int = 5

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        colors = random.sample(range(10), n_inputs)
        return [create_img(shape, color=color) for shape, color in zip(shapes, colors)]
    
    def create_code(self, inputs):
        """
        Uses the shape of the input images to create drawings:
        - vertical and horizontal lines
        - lines, rectangles, and pixels referenced to the shape of the image
        """
        n_draws = random.randint(self.min_draws, self.max_draws)
        code = ''
        min_shape = np.min([img.shape for img in inputs], axis=0)

        function_names = [
            "draw_vertical_line",
            "draw_horizontal_line",
            "draw_line",
            "draw_rectangle",
            "draw_pixel"
        ]


        for _ in range(n_draws):
            function_name = random.choice(function_names)
            # TODO: add more shape dependent drawings
            if function_name == "draw_vertical_line":
                # vertical line
                x = random.choice(list(range(0, min_shape[1]//2)) + list(range(-min_shape[1]//2, 0)))
                code += f"draw_vertical_line(img, x={x}, color={random.randint(0, 9)})\n"
            elif function_name == "draw_horizontal_line":
                # horizontal line
                y = random.choice(list(range(0, min_shape[0]//2)) + list(range(-min_shape[0]//2, 0)))
                code += f"draw_horizontal_line(img, y={y}, color={random.randint(0, 9)})\n"
            elif function_name in ["draw_line", "draw_rectangle"]:
                x1 = random.choice([0, 'img.shape[1] // 2'])
                if x1 == 'img.shape[1] // 2':
                    x2 = 'img.shape[1] - 1'
                else:
                    x2 = random.choice(['img.shape[1] // 2', 'img.shape[1] - 1'])
                y1 = random.choice([0, 'img.shape[0] // 2'])
                if y1 == 'img.shape[0] // 2':
                    y2 = 'img.shape[0] - 1'
                else:
                    y2 = random.choice(['img.shape[0] // 2', 'img.shape[0] - 1'])
                color = random.randint(0, 9)
                code += f"{function_name}(img, point1=({y1}, {x1}), point2=({y2}, {x2}), color={color})\n"
            elif function_name == "draw_pixel":
                x = random.choice([0, 'img.shape[1] // 2', 'img.shape[1] - 1'])
                y = random.choice([0, 'img.shape[0] // 2', 'img.shape[0] - 1'])
                color = random.randint(0, 9)
                code += f"draw_pixel(img, point=({y}, {x}), color={color})\n"

        code = wrap_code_in_function(code)
        return code
                

@dataclass
class GeometricTransformations(TrainingTask):
    min_inputs: int = 2
    max_inputs: int = 5
    min_side: int = 3
    max_side: int = 10
    min_transformations: int = 1
    max_transformations: int = 4

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        if random.random() < 0.5:
            return [Img(np.random.randint(0, 10, size=shape)) for shape in shapes]
        else:
            return [create_image_with_random_objects(shape) for shape in shapes]

    def create_code(self, inputs):
        parameter_functions = [
            random_rotate_90_parameters,
            random_flip_parameters
        ]
        parameter_functions.append(random_upscale_parameters)
        if random.random() < 0.5:
            parameter_functions.append(random_pad_parameters)
        else:
            parameter_functions.append(random_trim_parameters)
        random.shuffle(parameter_functions)
        n_transformations = random.randint(self.min_transformations, self.max_transformations)
        parameter_functions = parameter_functions[:n_transformations]

        code = ''
        outputs = inputs
        for parameter_function in parameter_functions:
            parameters = parameter_function(outputs)
            function_name = parameter_function.__name__.replace("random_", "").replace("_parameters", "")
            new_line = f"img = {function_name}(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
            code += new_line
            outputs = safe_code_execution(wrap_code_in_function(new_line), outputs)
        code = wrap_code_in_function(code)
        return code


@dataclass
class Downscale(TrainingTask):
    min_inputs: int = 2
    max_inputs: int = 5
    min_side: int = 2
    max_side: int = 5
    min_upscale: int = 2
    max_upscale: int = 5

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        if random.random() < 0.5:
            inputs = [create_image_with_random_objects(shape) for shape in shapes]
        else:
            inputs = [Img(np.random.randint(0, 10, size=shape)) for shape in shapes]
        scale = random.randint(self.min_upscale, self.max_upscale)
        inputs = [upscale(img, scale=(scale, scale)) for img in inputs]
        return inputs

    def create_code(self, inputs):
        parameters = random_downscale_parameters(inputs)
        code = f"img = downscale(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code = wrap_code_in_function(code)
        return code


@dataclass
class LearnDetectObjectsParameters(TrainingTask):
    min_inputs: int = 3
    max_inputs: int = 5
    min_side: int = 3
    max_side: int = 10

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        inputs = [create_image_with_random_objects(shape) for shape in shapes]
        if random.random() < 0.33:
            new_background_color = random.randint(1, 9)
            colormap = {0: new_background_color, new_background_color: 0}
            inputs = [apply_colormap(img, colormap) for img in inputs]
            metadata = dict(background_color=new_background_color)
        else:
            metadata = dict(background_color=0)
        return inputs, metadata

    def create_code(self, inputs, metadata):
        # TODO: better control for connectivity and monochrome
        parameters = dict(connectivity=random.choice([4, 8]), monochrome=random.choice([True, False]),
                          **metadata)
        code = f"objects = detect_objects(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code += "n = len(objects)\n"
        code += f"output = create_img((n, n), color={random.randint(0, 9)})\n"
        code += 'return output\n'
        code = wrap_code_in_function(code)
        return code
    

@dataclass
class ChangeObjectColorBasedOnArea(LearnDetectObjectsParameters):
    min_inputs: int = 3
    max_inputs: int = 5
    min_side: int = 8
    max_side: int = 10
    n_objects: int = 5
    # TODO: add more variability on the sizes

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        metadata = dict(allowed_sizes=[2, 3, 4], n_objects=self.n_objects, connectivity=random.choice([4, 8]),
                        monochrome=random.choice([True, False]),
                        background_color=random.choice([0]*18 + list(range(1, 10))))
        inputs = [generate_arc_image_with_random_objects(shape, **metadata)[0] for shape in shapes]
        return inputs, metadata

    def create_code(self, inputs, metadata):
        allowed_sizes = metadata.pop('allowed_sizes')
        metadata.pop('n_objects', None)  # n_objects is not used in this task
        parameters = dict(**metadata)
        code = f"objects = detect_objects(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code += f"output = create_img(img.shape, color={metadata['background_color']})\n"
        new_colors = random.sample([color for color in range(10) if color != metadata['background_color']], len(allowed_sizes))
        area_to_color = {size: color for size, color in zip(allowed_sizes, new_colors)}
        code += f'area_to_color = {area_to_color}\n'
        code += 'for object in objects:\n'
        code += '    object.change_color(area_to_color[object.area])\n'
        code += '    draw_object(output, object)\n'
        code += 'return output\n'
        code = wrap_code_in_function(code)
        return code


@dataclass
class ChangeObjectColorBasedOnHeightOrWidth(LearnDetectObjectsParameters):
    min_inputs: int = 3
    max_inputs: int = 5
    min_side: int = 8
    max_side: int = 10
    n_objects: int = 7
    allowed_size: int = 2
    # TODO: add more allowed sizes

    def create_inputs(self):
        n_inputs = random.randint(self.min_inputs, self.max_inputs)
        shapes = [np.random.randint(self.min_side, self.max_side + 1, 2) for _ in range(n_inputs)]
        metadata = dict(allowed_sizes=[self.allowed_size], n_objects=self.n_objects, connectivity=random.choice([4, 8]),
                        monochrome=random.choice([True, False]),
                        background_color=random.choice([0]*18 + list(range(1, 10))))
        inputs = [generate_arc_image_with_random_objects(shape, **metadata)[0] for shape in shapes]
        return inputs, metadata

    def create_code(self, inputs, metadata):
        metadata.pop('allowed_sizes')
        metadata.pop('n_objects', None)  # n_objects is not used in this task
        parameters = dict(**metadata)
        code = f"objects = detect_objects(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code += f"output = create_img(img.shape, color={metadata['background_color']})\n"

        property = random.choice(['height', 'width'])
        new_colors = random.sample([color for color in range(10) if color != metadata['background_color']], self.allowed_size)
        colormap = {key: color for key, color in zip(range(1, len(new_colors) + 1), new_colors)}
        code += f'{property}_to_color = {colormap}\n'
        code += 'for object in objects:\n'
        code += f'    object.change_color({property}_to_color[object.{property}])\n'
        code += '    draw_object(output, object)\n'
        code += 'return output\n'
        code = wrap_code_in_function(code)
        return code


#TODO: use object properties (is_line, point, rectangle, etc.) to change colors, move or filter.
#TODO: colormap over random images
#TODO: task with changing background color, how to find background color?
#TODO: move objects based on some object property