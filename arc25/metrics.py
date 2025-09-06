import numpy as np
import pandas as pd


def pixel_similarity_score(ground_truth: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute a pixel-wise similarity score between two 2D integer matrices.

    - If the matrices have the same shape, returns the fraction of pixels that match.
    - If the shapes differ, finds the best overlap alignment and divides
      the number of matching pixels by the area of the smallest bounding box
      that contains both matrices.

    Returns a float in [0.0, 1.0].
    """
    gt = np.asarray(ground_truth)
    ref = np.asarray(reference)

    h1, w1 = gt.shape
    h2, w2 = ref.shape

    # same shape: simple pixel accuracy
    if h1 == h2 and w1 == w2:
        return float(np.mean(gt == ref))

    # different shapes: slide one over the other to maximize matches
    bbox_h = max(h1, h2)
    bbox_w = max(w1, w2)
    denom = bbox_h * bbox_w
    best_matches = 0

    # dx, dy are offsets from ref to gt
    for dx in range(-(h2 - 1), h1):
        for dy in range(-(w2 - 1), w1):
            # overlap region in gt
            i1_start = max(0, dx)
            i1_end   = min(h1, h2 + dx)
            j1_start = max(0, dy)
            j1_end   = min(w1, w2 + dy)
            if i1_end <= i1_start or j1_end <= j1_start:
                continue

            # corresponding region in ref
            i2_start = i1_start - dx
            i2_end   = i1_end   - dx
            j2_start = j1_start - dy
            j2_end   = j1_end   - dy

            region_gt  = gt[i1_start:i1_end, j1_start:j1_end]
            region_ref = ref[i2_start:i2_end, j2_start:j2_end]
            matches = np.count_nonzero(region_gt == region_ref)

            if matches > best_matches:
                best_matches = matches

    return best_matches / float(denom)


def correct_grids_score(outputs, preds):
    scores = []
    for output, pred in zip(outputs, preds):
        if output.shape != pred.shape: # TODO: this does not work with my Img implementation
            scores.append(0.0)
        else:
            if np.all(output == pred):
                scores.append(1.0)
            else:
                scores.append(0.0)
    return np.mean(scores)


def aggregate_metrics(results):
    df = pd.DataFrame()
    for task_id, task_results in results.items():
        n_preds = len(task_results)
        df.loc[task_id, 'n_preds'] = n_preds
        df.loc[task_id, 'valid code'] = (len([1 for result in task_results if 'code' in result]))/n_preds
        df.loc[task_id, 'valid outputs'] = (len([1 for result in task_results if 'error_type' not in result]))/n_preds
        df.loc[task_id, 'unique outputs'] = len(set(result['fingerprint'] for result in task_results if 'fingerprint' in result))/n_preds
        for partition in ['train', 'test']:
            df.loc[task_id, f'{partition}_pixel_score'] = np.mean([result.get(f'{partition}_pixel_score', 0) for result in task_results])
            df.loc[task_id, f'{partition}_correct_grids'] = np.mean([result.get(f'{partition}_correct_grids', 0) for result in task_results])
            df.loc[task_id, f'{partition}_pass_rate'] = sum(result.get(f'{partition}_is_correct', 0) for result in task_results)/n_preds
            df.loc[task_id, f'{partition}_is_correct'] = int(any(result.get(f'{partition}_is_correct', 0) for result in task_results))
    if 'test_is_correct' in df.columns:
        df['is_correct'] = df['train_is_correct'] * df['test_is_correct']
    df.loc['MEAN'] = df.mean(axis=0)
    return df.astype(float)


def error_analysis(results):
    errors_to_check = ['TimeoutException', 'NonDeterministicCode', 'UnsafeCode', 'ParsingCodeFailed']

    df = pd.DataFrame(columns=['n_preds', 'error_rate'] + errors_to_check)
    all_errors = []
    for task_id, task_results in results.items():
        task_errors = [result['error_type'] for result in task_results if 'error_type' in result]
        all_errors.extend(task_errors)
        df.loc[task_id, 'n_preds'] = len(task_results)
        df.loc[task_id, 'error_rate'] = len(task_errors) / len(task_results) if task_results else 0.0
        for error_type in errors_to_check:
            df.loc[task_id, error_type] = sum(1 for error in task_errors if error == error_type) / len(task_results) if task_results else 0.0
    df.loc['MEAN'] = df.mean(axis=0)

    error_counts = pd.Series(all_errors).value_counts()
    print("Most common errors:")
    print(error_counts.head(20))
    return df.astype(float)

