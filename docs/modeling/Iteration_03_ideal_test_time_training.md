# Iteration 3. Ideal test-time training setup

_07-04-2025_

## Goal

Update the architects' code to be able to make training and inference for each task.

## Motivation

ARC tasks are independent, thus when doing test-time training is better to focus on each task instead of training on all the tasks at the same time. Knowledge transfer between the different tasks should be very small, so fine-tuning a custom model for each task should be the best strategy.

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

### Increasing GPU usage

After looking at the plots of GPU usage I have noticed that I could increase the number of slots per GPU both on training and inference.

| train GPU slots | inf GPU slots | mean GPU usage | max VRAM | training time (s) | inference time (s) |
|-----------------|---------------|----------------|----------|-------------------|--------------------|
| 1               | 2             | 89.5%          | 51.4%    | 7087              | 7360               |
| 2               | 2             | 93.9%          | 50.2%    | 6864              | 6748               |
| 2               | 3             | 95.0%          | 76.6%    | 6962              | 6902               |

The most reliable metric is mean GPU usage. Inference time we already know that it is not reliable and there is
some variability on training times due to the random assignment of the tasks. Using 2 slots per GPU for training and
3 for inference should give a speedup of around 6%, which is 43 minutes for the 12 hour run. Not game changing but very welcome.

[Link to full results](https://docs.google.com/spreadsheets/d/1NmmCZA7gPOyoBypwvpw_JhYdjcvqNFHibX_WahwTHIM/edit?gid=0#gid=0&range=A42)

## Results

### Evaluation vs test set

On this [notebook](https://www.kaggle.com/code/ironbar/the-architects-single-task-ttt/notebook) I have run the exact same setup that has scored 10.17 on the leaderboard and took 9 hours to run. 

If I run the exact same configuration on the evaluation set it only takes 4 hours and scores 10.6 (I'm not sure what the architects prints mean because on them the score is 8.7).

The difference in speed is caused because when we are doing the submission the system is evaluated against both partitions of the test set, so that is 240 tasks instead of 120. So I don't have to worry about my system doing timeout on the private test set because the system has already done predictions for it.

### Training epochs

![alt text](res/1745469291160_image.png)

It seems that a small number of training epochs (6) is bad, but also once we reach a certain number of epochs (8-10) increasing the training length is not beneficial. Maybe I have to lower the learning rate when using a bigger number of epochs?

### Train Max Sequence Length

![alt text](res/1745469708409_image.png)

The tendency is not very clear, but the best results are obtained when using 8192 which is the maximum training sequence length available for the current model.

Submission time increases slightly.

### Lora rank

![alt text](res/1745469940269_image.png)

It might seem that using a bigger lora rank is beneficial.

### Uncertainty on the LB results

Let's submit the same configuration, just changing the random seed.

TODO:

### Learning rate

I might get better results with a lower learning rate and longer training?

TODO:

## Conclusion

## Next steps

## TODO

- [x] How GPU usage looks when using batch size 1?
- [x] What if I copy the base model to `dev/shm`
- [ ] Tune the submission hyperparameters
  - [x] Lora rank
  - [x] Number of training epochs (better change epochs than learning rate when possible)
  - [ ] Inference parameters (n and min_prob)
  - [ ] Learning rate
  - [ ] My intuition is that I should train as long as possible, and make just 8 predictions per task.
  - [ ] Uncertainty on the results (what if I change the random seed?)
  - [x] Are the training samples correctly sorted? Maybe they are not optimal for single task training. The order is random.
- [x] Check the evaluation prints of the architects. They are different to normal scoring
- [ ] Make more evaluations on the evaluation set and compare to test set. I want to see a correlation of runtime and score.
- [x] What if I use 2 GPU slots for training? Currently just 40% of GPU memory is used.