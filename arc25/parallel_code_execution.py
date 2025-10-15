import logging
import numpy as np
from typing import List
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from joblib.externals.loky.process_executor import TerminatedWorkerError
import hashlib

from arc25.code_execution import safe_code_execution
from arc25.prompting import parse_python_code_from_response
from arc25.metrics import get_metrics
from arc25.validation import validate_outputs
from arc25.data_augmentation import apply_data_augmentation, revert_data_augmentation

logger = logging.getLogger(__name__)


class CodeRunner():
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.parallel = None
        self.start_parallel()

    def start_parallel(self):
        self.parallel = Parallel(n_jobs=self.n_jobs, backend="loky", prefer="processes", batch_size='auto')

    def run(self, tasks, task_ids, text_predictions, data_augmentation_params,
            batch_size=5000, group_results_by_task=True, timeout_duration=1,
            disable_tqdm=False):
        results = []
        for i in tqdm(range(0, len(tasks), batch_size), desc="Executing predictions", unit="batch",
                    disable=len(tasks)<=batch_size or disable_tqdm, smoothing=0):
            batch = list(zip(text_predictions[i:i+batch_size], tasks[i:i+batch_size], task_ids[i:i+batch_size], data_augmentation_params[i:i+batch_size]))
            try:
                extra_kwargs = dict(timeout_duration=timeout_duration, execution_method='exec')
                with tqdm_joblib(total=len(batch), desc=f"Executing predictions for batch {i//batch_size} with exec",
                                unit="run", smoothing=0, disable=disable_tqdm):
                    results.extend(self.parallel(delayed(_run_one)(*args, **extra_kwargs) for args in batch))
            except TerminatedWorkerError:
                logger.warning("TerminatedWorkerError encountered with 'exec' method, retrying with 'subprocess' method.")
                extra_kwargs = dict(timeout_duration=timeout_duration, execution_method='subprocess')
                with tqdm_joblib(total=len(batch), desc=f"Executing predictions for batch {i//batch_size} with subprocess",
                                unit="run", smoothing=0, disable=disable_tqdm):
                    results.extend(self.parallel(delayed(_run_one)(*args, **extra_kwargs) for args in batch))
        if not group_results_by_task:
            return results
        grouped_results = {}
        for result in results:
            task_id = result.pop('task_id')
            if task_id not in grouped_results:
                grouped_results[task_id] = []
            grouped_results[task_id].append(result)
        return grouped_results


def _run_one(text_prediction, task, task_id, data_augmentation_params,
             timeout_duration=5, execution_method='exec'):
    code = parse_python_code_from_response(text_prediction)
    if not code:
        return dict(error_type="ParsingCodeFailed", error_message='', text_prediction=text_prediction,
                    task_id=task_id)
    try:
        input_grids = [sample['input'] for sample in task['train']] + [sample['input'] for sample in task['test']]
        if data_augmentation_params is not None:
            input_grids = apply_data_augmentation(input_grids, **data_augmentation_params)
        output_grids = safe_code_execution(
            add_additional_imports(remove_unnecessary_lines(code)),
            input_grids,
            func_name="transform",
            execution_method=execution_method,
            timeout_duration=timeout_duration,
        )
        output_grids = validate_outputs(output_grids)
        if data_augmentation_params is not None:
            original_output_grids = revert_data_augmentation(output_grids, **data_augmentation_params)
        else:
            original_output_grids = output_grids
        result = dict(code=code, output_grids=output_grids,
                      input_grids=input_grids, text_prediction=text_prediction,
                      fingerprint=fingerprint(original_output_grids),
                      test_output_grids=original_output_grids[-len(task['test']):],
                      task_id=task_id, data_augmentation_params=data_augmentation_params)
        result.update(get_metrics(task, original_output_grids))
        return result
    except BaseException as e:
        return dict(code=code, error_type=type(e).__name__, error_message=str(e), task_id=task_id)


def remove_unnecessary_lines(code):
    remove_line_keywords = ['print(', 'from common import *']
    code = '\n'.join(line for line in code.split('\n') if not any(keyword in line for keyword in remove_line_keywords))
    return code.strip()


def add_additional_imports(code):
    additional_imports = [
        'from typing import List, Tuple',
        'import numpy as np',
        'import numpy',
        'from arc25.BARC_dsl import *',
    ]
    imports = '\n'.join(additional_imports)
    return imports + '\n' + code if code else imports


def fingerprint(output_grids : List[np.ndarray]) -> str:
    """
    Create a compact hash for a list of matrices.
    Includes shape & dtype to distinguish e.g. (2×2) from (4×1).
    """
    h = hashlib.sha256()
    for grid in output_grids:
        # incorporate shape and dtype in a reproducible way
        h.update(str(grid.shape).encode())
        h.update(grid.dtype.str.encode())
        # raw data bytes
        h.update(grid.tobytes())
    return h.hexdigest()
