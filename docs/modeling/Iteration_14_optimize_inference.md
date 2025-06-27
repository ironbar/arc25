# Iteration 14. Optimize inference

_26-06-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Is there room for improvement with the current approach if I modify the inference to better explore
the solution space?

## Motivation

On [Iteration 12, solve a few arc tasks](Iteration_12_solve_a_few_arc_tasks.md) I saw very little
exploration of the solution space. I want to try different ideas to see if we can fix the problem
without redesigning the training and the strategy.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ] What is the max sequence length for training and inference on a GPU with 24GB of VRAM?
- [ ] Better sampling strategy. Could play with temperature, top_k and top_p to create more diverse samples. https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/text_generation#transformers.GenerationConfig.temperature
- [ ] What if I give hints of how to solve the problem int the prompt? Is the model capable on that case?
- [ ] What if I have a multi-turn conversation with the model to improve its own code?
