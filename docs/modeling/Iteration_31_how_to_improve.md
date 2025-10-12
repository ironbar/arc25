# Iteration 31. How to improve from 20% to 100%?

_12-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Think how to improve from 20% to 100% on the ARC-AGI-1 evaluation set.

## Motivation

So far we have verified that we can use hindsight relabel to adapt the model to new tasks and
increase the ARC score. However the BARC induction model is only solving ~20% of the ARC-AGI-1 evaluation
set, and around 1% of the ARC-AGI-2 evaluation set. This model is not capable of solving ARC with
the compute constraints of the submission.

We need to think outside the box to be able to make a dramatic improvement.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ] Analyze the predictions on the unsolved train and evaluation tasks. Are they in the right direction? Why the model is not solving them?
- [ ] Sample efficiency in the RL literature. https://chatgpt.com/share/68eb58e6-fe78-8012-b33c-cb0689f482c2
- [ ] Divide and conquer approach: What if we try to find programs that solve only a fraction of the task samples?
  I believe this could give a small boost but probably won't give the improvement we need.
- [ ] A smaller model with a long context could be trained with RL to learn to search. Instead of doing
  multiple independent predictions, the model would use information from previous predictions to
  either refine the approach or try completely different approaches. This should be more sample
  efficient than the current approach. And the training is more aligned with the goal. The first attempt
  uses 9500 tokens (8500 for task encoding and 1000 for the prediction), second attempt is cheaper at 5000 (4000 for grids and 1000 for code prediction).
  With a context size of 32k we could make 5 attempts in the worst case (4 train samples of 30x30).
- My current implementation only uses HER, maybe I should combine it with GRPO.
