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

### How humans solve abstraction and reasoning (ARC) tasks?

Humans have the core knowledge priors, so when we look at the pixels of an ARC grid we create an
abstract representation of the image. We can describe the grid with natural language, we can understand
what the input grids have in common, what the outputs have in common...
In summary, **abstraction** allows us to see not just a collection of pixels, but to build high level
abstractions that represent the images and allow to compare them and see what the differences are and
what do they have in common. 

Those abstractions help to reduce the search space when we try to solve a task. We describe the tasks
using natural language, describing the transformation between the inputs and the outputs. When we **reason**
we draw some hypothesis of what the task is about. Then we can use our internal world model to transform
the grids using the hypothesis and verify if the outputs are correct.
On every failed attempt we learn more about the task. In fact an intelligent person should make the attempts
that will maximize its learning. That eventually leads to find the right solution.

Observe, hypothesize, test, learn, repeat. Solving ARC requires to use the scientific method.

![](../res/how-humans-solve-arc.png)

### How my current system compares against a human?

I'm pretty sure my model does not have abstractions as powerful and general as humans. One way to enhance
the representations of the model would be teh omni-arc approach from last year ARC24 challenge.

Humans have **memory**. Memory allows to explore the search space without repeating previous errors.
My system does many predictions independently, and this produces repeated predictions.

Humans have the ability to correct/refine an incorrect solution. However the BARC induction model
that I'm currently using does not have this ability.

![](../res/how-ai-might-solve-arc.png)

### Python code and the python interpreter

I still believe the easiest way to solve ARC is to use python code and the python interpreter.

It is true that frontier models rely on natural language to describe the tasks and use their own
capabilities as a world model.

But for any task that can be described with natural language there should also be a python program
that implements the task. And I would argue that with the right DSL the python program should be
short and elegant. Thus I don't see the limitations of using python.

Finally the python interpreter always works and in contrast all models are fallible.

### Transduction

Transduction relies on the model to generate a good representation of the ARC tasks. At test time
the model can use in context learning or test-time training to adapt to new tasks.

Test-time training is crucial for transduction approaches. My solution improved from 11% to 33% just
by doing test-time training.

How could this approach be improved:

- Pre-train on more data
- Use an architecture with better inductive priors, that will better represent the programs.
- Improvements in the test-time training setup

Transduction can solve ARC, but I don't believe is the easiest way to do it. To be able to work
the model has to learn an internal representation almost equivalent to a python program. I believe
that transduction is more difficult than generating python code.

The biggest advantage of transduction is that it can use gradient descent at test-time to adapt to
the new tasks. In the other hand we don't have yet a similar adaptation mechanism when using induction.

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
