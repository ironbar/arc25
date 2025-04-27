import signal
import numpy as np

from arc25.dsl import *


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


def safe_code_execution(code, inputs, func_name='task', timeout_duration=1):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    restricted_locals = {}
    restricted_globals = globals() # TODO: restrict the globals

    # Dynamically define the function to be executed
    exec(code, restricted_globals, restricted_locals)
    if func_name not in restricted_locals:
        raise ValueError(f"The code did not define the expected '{func_name}' function.")

    func = restricted_locals[func_name]
    result = _generate_outputs(inputs, func)
    signal.alarm(0)
    return result


def _generate_outputs(inputs, func):
    if isinstance(inputs, list):
        return [func(input.copy()) for input in inputs]
    else:
        raise NotImplementedError("Currently only list of inputs is supported.")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution exceeded time limit!")