"""
DSL Training tasks

All training tasks return: inputs, outputs and code

Each training task should teach a specific concept, and the name of the task should reflect that
"""
import random
from abc import ABC, abstractmethod
from collections import namedtuple
from arc25.dsl import *
from arc25.input_generation import *


Task = namedtuple("Task", ["inputs", "outputs", "code"])


class TrainingTask(ABC):
    def sample(self):
        inputs = self.create_inputs()
        code = self.create_code(inputs)
        code = validate_code(code, inputs)
        outputs = apply_code(code, inputs)
        return Task(inputs=inputs, outputs=outputs, code=code)

    @abstractmethod
    def create_inputs(self):
        pass

    @abstractmethod
    def create_code(self, inputs):
        pass


class RandomDrawingTask(TrainingTask):
    def create_inputs(self):
        return [create_img(np.random.randint(3, 10, 2), color=random.randint(0, 9))]
    
    def create_code(self, inputs):
        n_draws = random.randint(1, 5)
        parameter_functions = [
            random_draw_line_parameters,
            random_draw_rectangle_parameters,
            random_draw_horizontal_line_parameters,
            random_draw_vertical_line_parameters,
            random_draw_pixel_parameters
        ]
        parameter_functions = [
            random_draw_line_parameters,
        ]
        code = ''
        for _ in range(n_draws):
            parameter_function = random.choice(parameter_functions)
            parameters = parameter_function(inputs[0])
            function_name = parameter_function.__name__.replace("random_", "").replace("_parameters", "") 
            code += f"{function_name}(img, {', '.join(f'{k}={v}' for k, v in parameters.items())})\n"
        code = wrap_code_in_function(code)
        return code
    

def wrap_code_in_function(code):
    function_code = "def task(img):\n"
    indent="    "
    for line in code.splitlines():
        function_code += f"{indent}{line}\n"
    function_code += f"{indent}return img"
    function_code += "\n"
    return function_code


def validate_code(code, inputs):
    # This function should validate the code and return a valid code
    # For now, we will just return the code as is
    return code

def apply_code(code, inputs):
    # This function should apply the code to the inputs and return the outputs
    # For now, we will just return the inputs as is
    # img = inputs[0]
    # exec(code)
    return inputs