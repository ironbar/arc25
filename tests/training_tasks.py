"""
DSL Training tasks

All training tasks return: inputs, outputs and code

Each training task should teach a specific concept, and the name of the task should reflect that
"""
import random
from abc import ABC, abstractmethod
from collections import namedtuple
from arc25.dsl import *


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
        pipeline = []
        