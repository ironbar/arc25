import pytest

from arc25.submission import sort_predictions_with_majority_voting_and_code_length

def test_sort_predictions_with_majority_voting_and_code_length_returns_most_voted_task():
    task_results = [
        {
            'original_output_grids': [[[1]], [[2]], [[1]]],
            'code': 'print(1)',
        },
        {
            'original_output_grids': [[[3]], [[2]], [[1]]],
            'code': 'print(2)',
        },
        {
            'original_output_grids': [[[3]], [[2]], [[4]]],
            'code': 'print(3)',
        },
        {
            'original_output_grids': [[[5]], [[6]], [[7]]],
            'code': 'print(4)',
        },
    ]
    sorted_predictions = sort_predictions_with_majority_voting_and_code_length(task_results, n_test=3)
    assert sorted_predictions[0][0]['pred'] == [[3]]
    assert sorted_predictions[1][0]['pred'] == [[2]]
    assert sorted_predictions[2][0]['pred'] == [[1]]


def test_sort_predictions_with_majority_voting_and_code_length_returns_shortest_task_when_tied():
    task_results = [
        {
            'original_output_grids': [[[1]], [[2]], [[3]]],
            'code': 'print()',
        },
        {
            'original_output_grids': [[[3]], [[4]], [[3]]],
            'code': 'print(2)',
        },
        {
            'original_output_grids': [[[4]], [[4]], [[4]]],
            'code': 'print(3)',
        },
        {
            'original_output_grids': [[[5]], [[6]], [[7]]],
            'code': 'print(4)',
        },
    ]
    sorted_predictions = sort_predictions_with_majority_voting_and_code_length(task_results, n_test=3)
    assert sorted_predictions[0][0]['pred'] == [[1]]
    assert sorted_predictions[1][1]['pred'] == [[2]]

    task_results = [
        {
            'original_output_grids': [[[1]], [[2]], [[1]]],
            'code': 'print(2)',
        },
        {
            'original_output_grids': [[[3]], [[4]], [[3]]],
            'code': 'print()',
        },
        {
            'original_output_grids': [[[4]], [[4]], [[4]]],
            'code': 'print(3)',
        },
        {
            'original_output_grids': [[[5]], [[6]], [[7]]],
            'code': 'print(4)',
        },
    ]
    sorted_predictions = sort_predictions_with_majority_voting_and_code_length(task_results, n_test=3)
    assert sorted_predictions[0][0]['pred'] == [[3]]
    assert sorted_predictions[2][0]['pred'] == [[3]]
