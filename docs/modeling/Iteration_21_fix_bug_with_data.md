# Iteration 21. Fix bug with data

_23-08-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

How good is the BARC induction model on the different ARC datasets?

## Motivation

I have discovered that I wasn't using the test samples when evaluating the BARC model. This make the problem harder in a way (because not all the training samples were given) and easier in another way (maybe the test samples are more difficult or cover some edge cases). On this iteration I need to stablish a good baseline so I can later check if test-time adaptation improves the scores.

I already know that data augmentation is helpful, so I will be using it by default on this iteration.

## Development

## Results

TODO: add image

| dataset              | n_preds | valid code | valid outputs | unique outputs | pixel similarity | correct grids | train_pass_rate | train_pass@n | pass_rate | pass@n |
|----------------------|---------|------------|---------------|----------------|------------------|---------------|-----------------|--------------|-----------|--------|
| training-arc-agi-1   | 240     | 100.00%    | 81.42%        | 43.10%         | 61.40%           | 14.66%        | 12.41%          | 61.75%       | 12.17%    | 61.25% |
| evaluation-arg-agi-1 | 464     | 100.00%    | 73.62%        | 45.19%         | 56.56%           | 2.85%         | 1.98%           | 23.00%       | 1.95%     | 22.25% |
| evaluation-arg-agi-2 | 264     | 100.00%    | 71.29%        | 51.58%         | 50.43%           | 0.11%         | 0.07%           | 0.83%        | 0.06%     | 0.83%  |

To be able to solve ARC, we need a model that has the right intuitions about how to solve a task. That seems to be the case for the ARC-AGI-1 datasets where we see a constant improvement when making more predictions. But the dynamics for the ARC-AGI-2 dataset are different. 
It is worth mentioning that it seems that we would need around 32768 to solve the training set. That would be around 4 hours per task (a prediction takes 0.4 seconds). So even for the training set the resources allowed in the Kaggle submission are not enough.

These numbers are very similar to the ones that I have previously to solving the bug.

One good property of this model is that the pass rate of the task is almost identical to the pass rate of just the training samples. This implies that a function that is able to solve the training samples is very likely to solve also the test samples.

## Conclusion

## Next steps

- I need to experiment with test-time training methods to see if I'm able to improve this metrics
- We need to increase the prediction efficiency of this model. This would be very likely achieved with reinforcement learning.
- I'm not exploiting the interactive nature of programming. The model should refine the code using execution feedback. Maybe the model can do this natively, but very likely needs also training.
- It is possible that ARC-AGI-2 data is out of distribution, and maybe we should label all ARC-AGI-2 tasks and generate new data a la BARC to be able to solve those tasks.

## TODO

- [ ] Visualize the solved tasks
