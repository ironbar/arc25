"""
DSL Training tasks

All training tasks return: inputs, outputs and code

Each training task should teach a specific concept, and the name of the task should reflect that
"""
import random
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


@dataclass
class RandomDrawingTaskOnEmptyImg(TrainingTask):
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
class RandomDrawingTaskOnEmptyImgs(RandomDrawingTaskOnEmptyImg):
    n_inputs = 2


@dataclass
class RandomDrawingTaskOnRandomImgs(RandomDrawingTaskOnEmptyImg):
    n_inputs = 3

    def create_inputs(self):
        shape = np.random.randint(self.min_side, self.max_side + 1, 2)
        return [Img(np.random.randint(0, 10, size=shape)) for _ in range(self.n_inputs)]


@dataclass
class RandomGeometricTransformations(TrainingTask):
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
#TODO: a generator that samples from all the tasks