from arc25.training_tasks import training_tasks_generator, _get_all_training_classes, Task

def test_sample_all_training_tasks():
    training_classes = _get_all_training_classes()
    for training_class in training_classes:
        task_generator = training_class()
        assert isinstance(task_generator.sample(), Task)


def test_training_tasks_generator():
    generator = training_tasks_generator()
    for _ in range(3):
        next(generator)
