import numpy as np


def create_submission(results: dict, sorting_metric: str = 'train_correct_grids') -> dict:
    submission = {}
    for task_id, task_results in results.items():
        n_test = len(dataset[task_id]['test'])
        valid_results = [result for result in task_results if 'original_output_grids' in result]
        if not valid_results:
            submission[task_id] = [{'attempt_1': [], 'attempt_2': [],} for _ in range(n_test)]
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
        # TODO: what to do if there aren't 2 attempts?
        task_submission = [{f'attempt_{idx}': value['pred']} for attempt in task_submission for idx, value in enumerate(attempt, 1)]
        submission[task_id] = task_submission


def sort_predictions_with_majority_voting_and_code_length(task_results, n_test):
    sorted_predictions = []
    for idx in range(n_test):
        unique_predictions = dict()
        for result in task_results:
            prediction = result['original_output_grids'][-n_test + idx]
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
