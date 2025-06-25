# Iteration 13. Reflections

_23-06-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Think about the current approach and propose next steps.

## Analysis of the current problems

## Thoughts about ARC

To solve new tasks we need to explore, learn and exploit the learning. The connection with reinforcement learning is obvious.

Let's analyze the two main approaches from ARC24.

### Test-time training. Transduction

Given enough input-output pairs, we can train a model to learn the mapping. With deep learning this usually
requires a big dataset or a pretrained model (or both). Generalization to new input samples is not guaranteed.

This is similar to system 1, intuition. To solve a problem very fast without explaining the solution, just having
a glance at the data and predicting the new output.

This approach can be improved by:

- Pretraining the model on more data
- Adding inductive biases to the model (so it learns faster the new tasks)

This approach is less likely to work on complex tasks that require multiple steps, or rules interacting with each other.
Thus it does not seem the most promising way to solve ARC-AGI-2.

### Search

It seems that o3 does search in the space of natural language programs. It describes the task with
natural language and generates the output using that description. It tries different approaches
for the same task and answers with the most promising solution.
We know that o3 was trained with reinforcement learning on the ARC training set. This RL trained
gave o3 the abilities to explore different solutions to the tasks and to select the one that is most likely correct.

This approach has some problems:

- Since it generates the grids directly, it can make errors despite describing the tasks correctly. There are not correction guarantees.

### The bitter lesson

Search and learn, the two methods that can scale arbitrarily with compute, noticed brilliantly on [Richard Sutton blogpost](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

Search and learn are the two methods that allow to adapt to novelty. We can search a program to do a new task,
or we can learn the new task directly.

### Synthesis: Search and learn

Limited to the expresivity of the DSL, but same applies to o3. It cannot do the actions that weren't described
on the training set.

## Motivation

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ]
