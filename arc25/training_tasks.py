"""
DSL Training tasks

All training tasks return: inputs, outputs and code

Each training task should teach a specific concept, and the name of the task should reflect that
"""
import random
import sys
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import namedtuple
from arc25.dsl import *
from arc25.input_generation import *
from arc25.code_execution import safe_code_execution, validate_code, wrap_code_in_function, InvalidCode

logger = logging.getLogger(__name__)


Task = namedtuple("Task", ["inputs", "outputs", "code", 'name'])


class TrainingTask(ABC):
    def sample(self, n_tries=3):
        inputs = self.create_inputs()
        for _ in range(n_tries):
            try:
                # TODO: better handling, what happens if we reach n_tries?
                code = self.create_code(inputs)
                code = validate_code(code, inputs)
                break
            except InvalidCode as e:
                logger.debug(f"{e}:\n{code}\nRetrying...")
                pass
            except Exception as e:
                logger.error(f"Unexpected error: {e}:\nRetrying...")
                pass
        outputs = safe_code_execution(code, inputs)
        return Task(inputs=inputs, outputs=outputs, code=code, name=self.__class__.__name__)

    @abstractmethod
    def create_inputs(self):
        pass

    @abstractmethod
    def create_code(self, inputs):
        pass


def training_tasks_generator():
    current_module = sys.modules[__name__]
    training_classes = [
        cls for name, cls in inspect.getmembers(current_module, inspect.isclass)
        if issubclass(cls, TrainingTask)
        and cls is not TrainingTask
        and cls.__module__ == __name__
    ]
    training_tasks = sorted([cls() for cls in training_classes], key=lambda x: x.__class__.__name__)
    logger.info(f"Found {len(training_tasks)} training tasks: {[task.__class__.__name__ for task in training_tasks]}")
    while True:
        # TODO: in the future I would like to be able to give weighted probabilities to the tasks
        task = random.choice(training_tasks)
        yield task.sample()


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
        for _ in range(n_draws):
            # TODO: add more shape dependent drawings
            if random.random() < 0.5:
                # vertical line
                x = random.choice(list(range(0, min_shape[1]//2)) + list(range(-min_shape[1]//2, 0)))
                code += f"draw_vertical_line(img, x={x}, color={random.randint(0, 9)})\n"
            else:
                # horizontal line
                y = random.choice(list(range(0, min_shape[0]//2)) + list(range(-min_shape[0]//2, 0)))
                code += f"draw_horizontal_line(img, y={y}, color={random.randint(0, 9)})\n"
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
        return [Img(np.random.randint(0, 10, size=shape)) for shape in shapes]

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
        inputs = [Img(np.random.randint(0, 10, size=shape)) for shape in shapes]
        scale = random.randint(self.min_upscale, self.max_upscale)
        inputs = [upscale(img, scale=(scale, scale)) for img in inputs]
        return inputs

    def create_code(self, inputs):
        parameters = random_downscale_parameters(inputs)
        code = f"img = downscale(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code = wrap_code_in_function(code)
        return code

#TODO: RandomDrawingTaskOnStructuredImg
#TODO: shape dependent drawings. Use references to the shape of the image to create the drawings