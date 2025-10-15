import signal
import numpy as np
import logging
import sys
import os
import pickle
import subprocess
from typing import Optional, List
from types import ModuleType
from contextlib import redirect_stdout, redirect_stderr
import io

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

logger = logging.getLogger(__name__)


def wrap_code_in_function(code):
    function_code = "def task(img):\n"
    indent="    "
    lines = code.strip().splitlines()
    for line in lines:
        function_code += f"{indent}{line}\n"
    if 'return ' not in lines[-1]:
        function_code += f"{indent}return img"
    function_code += "\n"
    return function_code


def validate_code(code, inputs, do_remove_irrelevant_lines=True):
    """
    1. Validates that the code is safe and deterministic
    2. Verifies that the task is meaningful and valid
    3. Removes irrelevant lines from the code
    """
    check_code_is_safe(code)
    check_code_is_deterministic(code)
    outputs =  safe_code_execution(code, inputs)
    check_at_least_one_output_is_different_to_input(inputs, outputs)
    check_all_outputs_are_valid(outputs)
    if do_remove_irrelevant_lines:
        code = remove_irrelevant_lines(code, inputs, outputs)
    return code


class UnsafeCode(Exception):
    pass


class NonDeterministicCode(Exception):
    pass


class InvalidCode(Exception):
    pass


def check_at_least_one_output_is_different_to_input(inputs, outputs):
    any_output_is_different = any(_is_different(input, output) for input, output in zip(inputs, outputs))
    if not any_output_is_different:
        raise InvalidCode("The code did not modify the input.")


def _is_different(x, y):
    return any(x.shape != y.shape) or not np.all(x == y)


def check_all_outputs_are_valid(outputs):
    all_outputs_are_valid = all(_is_valid_output(output) for output in outputs)
    if not all_outputs_are_valid:
        raise InvalidCode(f"The code did not produce valid outputs: {[output.shape for output in outputs]}.")


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
        try:
            temp_outputs = safe_code_execution(code_without_line, inputs)
            is_relevant_line = any(_is_different(temp_output, output) for temp_output, output in zip(temp_outputs, outputs))
        except Exception as e:
            #logger.debug(f"Error while checking relevance of line '{line}': {e}")
            # If an error occurs, we assume the line is relevant
            is_relevant_line = True
        if is_relevant_line:
            relevant_lines.append(line)
    validated_code = '\n'.join(relevant_lines)
    return validated_code


def check_code_is_safe(code):
    # TODO: verify that the findings are not in comment lines:
    # To make the output, replicate the 3x3 pattern to fill the entire grid by shifting it to the right and down by 3 pixels each time.
    forbidden_modules = ['logging', 'threading', 'bcrypt', 'datetime', 'os.sys', 'multiprocessing', 'time',
                         'os.path', 'pebble', 'hashlib', 'sys.exit', 'subprocess', 'calendar', 'os.environ',
                         'matplotlib', 'pygame']
    for module in forbidden_modules:
        if f'{module}.' in code or f'import {module}' in code:
            raise UnsafeCode(f"The code uses a forbidden module: {module}\nCode: {code}")

    forbidden_functions = ['input', '.save', 'write', 'open', 'exec', 'eval', 'compile']
    for func in forbidden_functions:
        if f'{func}(' in code:
            raise UnsafeCode(f"The code uses a forbidden function: {func}\nCode: {code}")

    forbidden_strings = ['except:']
    for s in forbidden_strings:
        if s in code:
            raise UnsafeCode(f"The code uses a forbidden string: {s}\nCode: {code}")



def check_code_is_deterministic(code):
    forbidden_modules = ['random']
    for module in forbidden_modules:
        if module in code:
            raise NonDeterministicCode(f"The code uses a forbidden module: {module}\nCode: {code}")
    # TODO: a more thorough check for non-deterministic code, running the code multiple times and checking if the output is the same


def _is_valid_output(output):
    return np.min(output.shape) >= 1 and np.max(output.shape) <= 30


def safe_code_execution(code: str, inputs: List[np.ndarray], func_name: str = 'task',
                        timeout_duration: int = 1, execution_method='exec', dsl: Optional[ModuleType] = None,
                        max_memory_mb: int = 2048):
    if execution_method == 'exec':
        return _safe_code_execution_exec(code, inputs, func_name, timeout_duration, dsl, max_memory_mb)
    elif execution_method == 'subprocess':
        return _safe_code_execution_subprocess(code, inputs, func_name, timeout_duration, dsl, max_memory_mb)
    else:
        raise ValueError(f"Unknown execution method: {execution_method}")


def _safe_code_execution_exec(code: str, inputs: List[np.ndarray], func_name: str = 'task',
                        timeout_duration: int = 1, dsl: Optional[ModuleType] = None, max_memory_mb: int = 2048):
    check_code_is_safe(code)
    check_code_is_deterministic(code)
    _set_memory_limit(max_memory_mb)
    _set_timeout_alarm(timeout_duration)
    namespace={'__builtins__': __builtins__, 'input_grids': inputs}
    if dsl is not None: namespace['dsl'] = dsl

    code = code + f'\n\noutput_grids = [{func_name}(input.copy()) for input in input_grids]'
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(code, namespace)
        return namespace['output_grids']
    except BaseException as e:
        logger.debug(f"Error during code execution: {e}")
        raise e
    except TimeoutException as e:
        logger.debug(f"Timeout during code execution: {e}")
        raise e
    finally:
        _disable_timeout_alarm()


def _set_timeout_alarm(timeout_duration):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)


def _disable_timeout_alarm():
    signal.alarm(0)


def _set_memory_limit(max_memory_mb: int):
    """Set memory limit for the current process (Linux/Unix only)."""
    if not HAS_RESOURCE:
        logger.warning("resource module not available, cannot set memory limit")
        return
    
    try:
        max_memory_bytes = max_memory_mb * 1024 * 1024
        # RLIMIT_AS limits the total address space (virtual memory)
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    except Exception as e:
        logger.warning(f"Failed to set memory limit: {e}")


class TimeoutException(BaseException):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution exceeded time limit!")


def _safe_code_execution_subprocess(
    code: str,
    inputs: List[np.ndarray],
    func_name: str = "task",
    timeout_duration: int = 2,
    dsl: Optional[ModuleType] = None,
    max_memory_mb: int = 2048,
):
    check_code_is_safe(code)
    check_code_is_deterministic(code)

    payload = {
        "inputs": inputs,
        "code": code,
        "func_name": func_name,
        "dsl_module_name": (dsl.__name__ if dsl is not None else None),
        "max_memory_mb": max_memory_mb,
    }

    launcher = r"""
import sys, pickle, importlib, numpy as np

try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

def set_memory_limit(max_memory_mb):
    if not HAS_RESOURCE:
        return
    try:
        max_memory_bytes = max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    except Exception:
        pass

def main():
    data = pickle.load(sys.stdin.buffer)
    inputs = data["inputs"]
    code = data["code"]
    func_name = data["func_name"]
    dsl_name = data["dsl_module_name"]
    max_memory_mb = data.get("max_memory_mb", 2048)

    # Set memory limit before executing user code
    set_memory_limit(max_memory_mb)

    # Keep a raw binary handle to real stdout for the pickle.
    raw_stdout = sys.stdout.buffer
    # Route user prints to stderr so they don't corrupt the pickle stream.
    sys.stdout = sys.stderr

    ns = {"__builtins__": __builtins__, "input_grids": inputs}
    if dsl_name:
        ns["dsl"] = importlib.import_module(dsl_name)

    exec(code, ns)
    fn = ns.get(func_name)
    if not callable(fn):
        raise RuntimeError(f"Function '{func_name}' not found or not callable")

    outputs = [fn(arr.copy()) for arr in inputs]
    pickle.dump(outputs, raw_stdout, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
"""

    # Launch the child in isolated mode (-I) and without site imports (-S).
    # start_new_session=True puts it in its own process group so we can kill everything it spawns.
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", launcher],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    try:
        stdout, stderr = proc.communicate(
            input=pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
            timeout=timeout_duration,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.wait()
        raise TimeoutException("Code execution exceeded time limit!")

    if proc.returncode != 0:
        err = stderr.decode("utf-8", "ignore").strip() or "Subprocess failed."
        raise RuntimeError(err)

    # stdout contains a clean pickle (no user prints).
    return pickle.loads(stdout)
