import numpy as np
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import hashlib

from arc25.code_execution import safe_code_execution
from arc25.prompting import parse_python_code_from_response
from arc25.metrics import get_metrics
from arc25.validation import validate_outputs
from arc25.data_augmentation import apply_data_augmentation, revert_data_augmentation


def run_code_from_predictions(tasks, task_ids, text_predictions, data_augmentation_params,
                              n_jobs=-1, batch_size=32000, group_results_by_task=True):
    work = list(zip(text_predictions, tasks, task_ids, data_augmentation_params))
    results = []
    for i in tqdm(range(0, len(work), batch_size), desc="Executing predictions", unit="batch"):
        batch = work[i:i+batch_size]
        with tqdm_joblib(total=len(batch), desc=f"Executing predictions for batch {i//batch_size}", unit="pred", smoothing=0):
            batch_results = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                prefer="processes",
                batch_size='auto', #1, 'auto'
            )(delayed(_run_one)(*args) for args in batch)
            results.extend(batch_results)
    if not group_results_by_task:
        return results
    grouped_results = {}
    for result in results:
        task_id = result.pop('task_id')
        if task_id not in grouped_results:
            grouped_results[task_id] = []
        grouped_results[task_id].append(result)
    return grouped_results


def _run_one(text_prediction, task, task_id, data_augmentation_params):
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
            execution_method='exec',
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
                      task_id=task_id)
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


def fingerprint(output_grids : list[np.ndarray]) -> str:
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
