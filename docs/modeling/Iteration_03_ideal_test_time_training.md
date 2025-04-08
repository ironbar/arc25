# Iteration 3. Ideal test-time training setup

_07-04-2025_

## Goal

Update the architects' code to be able to make training and inference for each task.

## Motivation

ARC tasks are independent, thus when doing test-time training is better to focus on each task instead of training on all the tasks at the same time.

I don't believe ARC can be solved using last year ARC24 solution, but being able to do test-time training for each task efficiently is very likely a part of this year solution.

## Development

### How the ideal solution looks like

1. We run 4 or 8 training processes in parallel, with batch size 1. Each training process would pick one remaining task, reset the PEFT model, train and save to disk.
2. We run 8 inference processes. Each inference process would pick one remaining task, load the PEFT, make inference and save results to disk. Ideally each process would do as many predictions as possible during the available time.

The unknown is how to load and unload the PEFT model efficiently. Every delay associated with changing the PEFT will be multiplied by 15 or 30 (depending if I use 8 or 4 processes.) So loading the model from disk, compiling... I need to find a way to do it really fast or even better don't have to do it.

### Time per task

We have 12 hours to solve 120 tasks. If we parallelize the system with 4 runs, that means we have 24 minutes per task.
So if doing inference for each task introduces an overhead of 1 minute per task, that still leaves 23 minutes per task.
So even a non efficient solution that wastes 1 minute to load and compile the model per task will have most of the time
for compute.

### Implementation

In this [notebook](https://www.kaggle.com/code/ironbar/the-architects-single-task-ttt) I have prepared an implementation that uses locks to select the GPU and the task.

Loading the model for training could take around 20s, for inference it is around 14s. So in total we could see a delay of around 30s per task, so aroun 15 minutes in total for a submission time of 12h hours, we can afford that.

### Base model in `dev/shm`

I have tried copying the model to `dev/shm` but did not observed any speedup. Probably when I read the model for the first
time it is cached. The model is slightly less than 4GB. [Notebook with experiments](https://www.kaggle.com/code/ironbar/the-architects-debug)

### Batch size and training speed

| experiment                 | batch_size=1 | batch_size=2 | batch_size=4 |
|----------------------------|--------------|--------------|--------------|
| 2 shortest tasks, 4 epochs | 72s          | 63s          | 58s          |
| 2 longest tasks, 1 epoch   | 125s         | 122s         | 136s         |

Clearly it pays to use a batch size of 1 if the gradient has enough information, that will allow to update the model more times.

### Comparison with my solution for ARC24 challenge

In my solution I could do 320 training steps for each task on ARC24 challenge. I was using a model of just 0.5B parameters versus the current 7B parameters. Now if I use 6 epochs that would be just 48 training steps, so training is 10 times shorter.

## Results

## Conclusion

## Next steps

## TODO

- [x] How GPU usage looks when using batch size 1?
- [x] What if I copy the base model to `dev/shm`
- [ ] Tune the submission hyperparameters
  - [ ] Lora rank
  - [ ] Number of training epochs
  - [ ] Inference parameters (n and min_prob)
  - [ ] Learning rate
  - [ ] My intuition is that I should train as long as possible, and make just 8 predictions per task.