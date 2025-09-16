from dataclasses import dataclass
import tyro

from arc25.utils import load_json

@dataclass
class Config:
    submission_path: str
    dataset_path: str


def main(args=None):
    config = tyro.cli(Config, description="Validate ARC submission")
    submission = load_json(config.submission_path)
    dataset = load_json(config.dataset_path)
    assert len(submission) == len(dataset), f"Submission has {len(submission)} tasks, dataset has {len(dataset)} tasks"
    for task_id in dataset.keys():
        assert task_id in submission, f"Task {task_id} not in submission"
        assert len(submission[task_id]) == len(dataset[task_id]['test']), f"Task {task_id} has {len(dataset[task_id]['test'])} test samples, submission has {len(submission[task_id])}"
        for sample in submission[task_id]:
            assert len(sample) == 2, f"Sample in task {task_id} does not have 2 elements"
            for key in ['attempt_1', 'attempt_2']:
                assert key in sample, f"Sample in task {task_id} does not have '{key}' key"
                grid = sample[key]
                assert isinstance(grid, list), f"Sample in task {task_id} '{key}' is not a list"
                assert all(isinstance(row, list) for row in grid), f"Sample in task {task_id} '{key}' is not a 2D list"
                assert all(isinstance(cell, int) for row in grid for cell in row), f"Sample in task {task_id} '{key}' does not contain all integers"
    print(f"Submission {config.submission_path} is valid for dataset {config.dataset_path}")


if __name__ == '__main__':
    main()
