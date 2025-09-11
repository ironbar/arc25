# Iteration 2x. Using RL to improve BARC induction model

_11-09-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I improve the BARC induction model using reinforcement learning?

## Motivation

I have read the [RL guide](https://docs.unsloth.ai/basics/reinforcement-learning-rl-guide) from unsloth
and they say that 300 samples are enough to see an improvement in the model. Probably I will need much
more compute for ARC but I would like to try.

The BARC induction model seems to have non-zero probability of solving the ARC-AGI-1 tasks, RL is
the way to increase that probability.

![alt text](res/1756274756683_image.png)

Ideas for the reward function:

- +1 if the model generates code
- +1 if the model generates running code
- Finally sum the ratio of correct grids. I believe that pixel accuracy is not a good metric, but I could try it also. I have the feeling that ARC is an all or nothing dataset, and pixel accuracy might lead to local optimums instead of leading to the global maximum.

On a first step I could try with a single training task. Then I could move to use all the training tasks.
I would measure the improvement on the training and the evaluation dataset. Finally if the technique
is helpful, I would move to using the synthetic dataset in a following iteration.

An additional motivation is that I have found that I would be able to make 512 predictions at maximum
for task on the Kaggle submission. That would solve just 22% of the ARC-AGI-1 evaluation
tasks. I need a model with a higher pass rate. RL is the way to get that.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ]
