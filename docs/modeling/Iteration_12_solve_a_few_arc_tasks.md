# Iteration 12. Solve a few ARC tasks

_17-06-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Probe that I can solve a few selected ARC tasks by using an LLM to write code.

## Motivation

On the previous [Iteration 10](Iteration_10_solve_arc_tasks.md) I tried to solve a few ARC tasks without
success: `08ed6ac7, 0b148d64, 0ca9ddb6, 0d3d703e, 178fcbfb, 1bfc4729, 1c786137`. The goal of this iteration
is to solve all those tasks by implementing new training tasks and/or improving the solving algorithm.

I should avoid creating training tasks that are clones from the real ARC tasks, otherwise I cannot
measure the generalization capability of the model. My goal should be to write training tasks that
teach the core knowledge that is needed for ARC.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ] Write new training tasks to solve the current knowledge gaps of the model
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.
- [ ] If possible I should use Kaggle compute for evaluation. It is almost free and is a good way to store and visualize results.
- [ ] Compositionality, can the model solve the task that selects the biggest object, crop and trim? That
  would be a good example of compositionality because those functions were not used together in the dataset
- [ ] Sequential solving. Try also solving the tasks in multiple steps, not just once. It could help
  with compositionality.
