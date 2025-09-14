import numpy as np


def create_submission(results: dict, dataset: dict, sorting_metric: str = 'train_correct_grids') -> dict:
    """
    Create a submission dictionary from results and dataset.
    Follows the ARC25 submission format.
    """
    submission = {}
    for task_id, task_results in results.items():
        n_test = len(dataset[task_id]['test'])
        valid_results = [result for result in task_results if 'test_output_grids' in result]
        if not valid_results:
            submission[task_id] = [{'attempt_1': [[0]], 'attempt_2': [[0]],} for _ in range(n_test)]
            continue
        scores = [result[sorting_metric] for result in valid_results]
        unique_scores = sorted(set(scores), reverse=True)
        task_submission = [[] for _ in range(n_test)]
        for score in unique_scores:
            results_with_score = [result for result in valid_results if result[sorting_metric] == score]
            sorted_predictions = sort_predictions_with_majority_voting_and_code_length(results_with_score, n_test)
            for idx in range(n_test):
                n_missing = 2 - len(task_submission[idx])
                if n_missing < 1:
                    continue
                task_submission[idx].extend(sorted_predictions[idx][:n_missing])

            if all(len(attempts) == 2 for attempts in task_submission):
                break
        # Ensure each case has 2 attempts
        for attempt in task_submission:
            if len(attempt) < 2:
                attempt.extend([{'pred': []}] * (2 - len(attempt)))
        formatted_task_submission = []
        for attempt in task_submission:
            formatted_task_submission.append({f'attempt_{idx}': value['pred'] for idx, value in enumerate(attempt, 1)})
        submission[task_id] = formatted_task_submission
    return submission


def sort_predictions_with_majority_voting_and_code_length(task_results, n_test):
    """
    Sort predictions based on majority voting and code length.

    Returns a list of length n_test, where each element is a list of unique predictions
    sorted by number of votes (descending) and mean code length (ascending).
    Each unique prediction is represented as a dictionary with keys: 'pred', 'votes', 'mean_code_length', 'code_lengths'.
    """
    sorted_predictions = []
    for idx in range(n_test):
        unique_predictions = dict()
        for result in task_results:
            prediction = result['test_output_grids'][idx]
            key = hash(tuple(map(tuple, prediction))) # str(prediction)
            if key not in unique_predictions:
                unique_predictions[key] = dict(pred=prediction, votes=0, code_lengths=[])
            unique_predictions[key]['votes'] += 1
            unique_predictions[key]['code_lengths'].append(len(result['code']))
        for unique_prediction in unique_predictions.values():
            unique_prediction['mean_code_length'] = float(np.mean(unique_prediction['code_lengths']))
        unique_predictions = sorted(unique_predictions.values(), key=lambda x: (-x['votes'], x['mean_code_length']))
        sorted_predictions.append(unique_predictions)
    return sorted_predictions


def evaluate_submission(ground_truth, submission):
    comparison = dict()
    for key, predictions in submission.items():
        comparison[key] = []
        solutions = ground_truth[key]
        for idx, solution in enumerate(solutions):
            comparison[key].append(any(prediction == solution for prediction in predictions[idx].values()))
    task_scores = {key: np.mean(values) for key, values in comparison.items()}
    print(f'Mean score: {np.mean(list(task_scores.values())):.1%}')
    task_above_zero = {key: values for key, values in task_scores.items() if np.mean(values) > 0}
    print(f'Tasks with non-zero score {len(task_above_zero)}: {task_above_zero}')
    return task_scores
