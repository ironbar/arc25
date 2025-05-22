import signal
import numpy as np
import logging

from arc25.dsl import *

logger = logging.getLogger(__name__)


def wrap_code_in_function(code):
    function_code = "def task(img):\n"
    indent="    "
    for line in code.splitlines():
        function_code += f"{indent}{line}\n"
    function_code += f"{indent}return img"
    function_code += "\n"
    return function_code


def validate_code(code, inputs):
    """
    Validate that the code modifies the input, generates a valid output and removes all irrelevant lines.
    """
    outputs =  safe_code_execution(code, inputs)
    check_at_least_one_output_is_different_to_input(inputs, outputs)
    check_all_outputs_are_valid(outputs)
    check_code_is_safe(code)
    check_code_is_deterministic(code)
    validated_code = remove_irrelevant_lines(code, inputs, outputs)
    return validated_code


def check_at_least_one_output_is_different_to_input(inputs, outputs):
    any_output_is_different = any(not np.all(input == output) for input, output in zip(inputs, outputs))
    if not any_output_is_different:
        raise InvalidCode("The code did not modify the input.")


def check_all_outputs_are_valid(outputs):
    all_outputs_are_valid = all(_is_valid_output(output) for output in outputs)
    if not all_outputs_are_valid:
        raise InvalidCode("The code did not produce valid outputs.")


def remove_irrelevant_lines(code, inputs, outputs):
    """
    Remove irrelevant lines from the code.

    TODO: could be dangerous to run when there are loops in the code.
    """
    lines = code.strip().split('\n')
    relevant_lines = []
    for idx, line in enumerate(lines):
        if idx == 0 or idx == len(lines) - 1:
            # Skip the first and last lines (function definition and return statement)
            relevant_lines.append(line)
            continue
        code_without_line = '\n'.join(lines[:idx] + lines[idx+1:])
        temp_outputs = safe_code_execution(code_without_line, inputs)
        is_relevant_line = any(not np.all(temp_output == output) for temp_output, output in zip(temp_outputs, outputs))
        if is_relevant_line:
            relevant_lines.append(line)
    validated_code = '\n'.join(relevant_lines)
    return validated_code


class UnsafeCode(Exception):
    pass


class NonDeterministicCode(Exception):
    pass


def check_code_is_safe(code):
    forbidden_modules = ['logging', 'threading', 'bcrypt', 'datetime', 'os.sys', 'multiprocessing', 'time',
                         'os.path', 'pebble', 'hashlib', 'sys.exit', 'subprocess', 'calendar', 'os.environ',]
    for module in forbidden_modules:
        if module in code:
            raise UnsafeCode(f"The code uses a forbidden module: {module}\nCode: {code}")


def check_code_is_deterministic(code):
    forbidden_modules = ['random']
    for module in forbidden_modules:
        if module in code:
            raise NonDeterministicCode(f"The code uses a forbidden module: {module}\nCode: {code}")
    # TODO: a more thorough check for non-deterministic code, running the code multiple times and checking if the output is the same


class InvalidCode(Exception):
    pass


def _is_valid_output(output):
    return np.min(output.shape) >= 1 and np.max(output.shape) <= 30


def safe_code_execution(code, inputs, func_name='task', timeout_duration=1):
    _set_timeout_alarm(timeout_duration)
    restricted_locals = {}
    restricted_globals = globals() # TODO: restrict the globals

    # Dynamically define the function to be executed
    try:
        exec(code, restricted_globals, restricted_locals)
    except Exception as e:
        logger.debug(f"Error during code execution: {e}")
        _disable_timeout_alarm()
        raise e
    # Check if the function is defined in the restricted locals
    if func_name not in restricted_locals:
        raise ValueError(f"The code did not define the expected '{func_name}' function.")

    func = restricted_locals[func_name]
    try:
        logger.debug(f"Executing code with function '{func_name}'")
        result = _generate_outputs(inputs, func)
        _disable_timeout_alarm()
        return result
    except Exception as e:
        logger.debug(f"Error during function execution: {e}")
        _disable_timeout_alarm()
        raise e


def _set_timeout_alarm(timeout_duration):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)


def _disable_timeout_alarm():
    signal.alarm(0)


def _generate_outputs(inputs, func):
    if isinstance(inputs, list):
        return [func(input.copy()) for input in inputs]
    else:
        raise NotImplementedError("Currently only list of inputs is supported.")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution exceeded time limit!")