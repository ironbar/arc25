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

## Results

## Conclusion

## Next steps

## TODO

- [ ] How GPU usage looks when using batch size 1?
- [ ] What if I copy the base model to `dev/shm`
