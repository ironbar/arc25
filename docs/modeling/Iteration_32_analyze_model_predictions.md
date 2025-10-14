# Iteration 32. Analyze model predictions

_12-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Analyze model predictions to understand the accuracy. Why is only solving ~20% of the ARC-AGI-1 evaluation tasks.

## Motivation

To be able to improve I need to understand why it does not solve the tasks.

## Development

Using predictions from previous experiments, I need to create a notebook to select the most accurate predictions and visualize them. I will do a random sampling of the unsolved tasks to diagnose the problems.

## Results

I have analyzed a random subset of 128 predictions, 16% of the evaluation ARC-AGI-1 tasks were solved.

![alt text](res/1760443849455_image.png)

The plot shows that model has a good intuition of ARC tasks. Only 20% are complete misunderstood.

But at the same time only 16% of the tasks are solved when doing 128 predictions per task. With 20k
predictions the solve rate is 38% according to the paper. But making so many predictions does
not have sense and it is not efficient. Making a few independent attempt makes sense to have diversity
in the predictions, but not in the order of hundreds or thousands.

## Conclusion

## Next steps

## TODO

- [ ]
