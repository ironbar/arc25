# Initial Plan

Since the end of ARC24 competition I have been thinking of how to better approach the challenge. Now that ARC25 has been launched I'm going to describe my initial ideas to try to solve ARC.

I have identified two possible paths that converge on the same approach.

**It's all about efficiency!**

## Path 1. Combine the best approaches: Induction and Test-time Training

Last year's competition showed that test-time training allowed the models to adapt to the novel tasks. At the same time in the semi-private dataset we saw that frontier models could generate code to solve more than half of the tasks.

Using code is a more promising approach because:

1. It is verifiable
2. Enables to iteratively refine the solution by comparing the outputs with the ground truth. I would argue that this is similar to reasoning.

My hypothesis is that we can use [hindsight experience replay (HER)](https://arxiv.org/abs/1707.01495) at test time to update the beliefs of the model and find the right solution more efficiently. Instead of sampling thousands of programs, sample a few and learn from the mistakes. **That is the way to combine induction and test-time training.**

We can treat the failed code attempts that run as new tasks, and train the model on those tasks. Those tasks will be in the neighborhood of the task that we want to solve.

We already know that HER enables faster learning, specially in very sparse reward environments.

![](res/2025-03-25-16-38-36.png)

## Path 2. Human inspired approach

When humans try to solve ARC tasks we draw some hypothesis and test it in our heads, if it is not correct we refine the hypothesis. To do this we use 3 modules:

- **Policy.** What action do I have to do to achieve the goal? Learned with hindsight
- **World model.** What happens if do this action? Learned with past experiences
- **Judgment.** Is the solution correct? Learned with human feedback or by comparison

Reasoning is iterative, we do it step by step combining the 3 modules above. 

My intuition is that o3 success in ARC-AGI-1 is likely due to an improved policy and better judgment. Vanilla LLMs are not good at judgment, but reasoning models need to be able to know if some answer is correct or wrong. By training with reinforcement learning the model improves its policy, it learns which strategies are good and which are bad to solve the ARC tasks. o3 very likely describes the task with natural language, generates the output grid and checks if the output looks correct. It might try different approaches, refine the description and when it is certain returns the response.

Focusing on efficiency the best configuration for ARC would be the following:

- Policy: model
- World model: python interpreter
- Judgement: metric function

That way we only have to learn the policy and we have guarantees that the other modules will be perfect.

## RL-ARC

The idea is to frame ARC as a reinforcement learning problem. The system is given a new task and it needs to learn it as efficiently as possible. It is like playing a game, but instead of hitting the buttons we have to write code.
The code generates an output that is evaluated against the ground truth and returns an score.

Finding the right program is equivalent to finding the right trajectory to solve a game. Instead of actions we write code, but the problem is exactly the same. When we want to solve a new task in ARC is the same as wanting to solve a new game. We can frame the problem as a Reinforcement learning game, with a very sparse reward.

The challenge is how to create a very efficient system to do this: how to design the DSL, how to pre-train the model, how to do split the compute between inference and test-time training... There is a huge number of possibilities that we need to explore
to find the winning system.

The DSL is perhaps one of the most critic parts of the system. Without a DSL the system will have to write
very long programs to solve the task, making the exploration of the solution space much harder and requiring
more compute (generating more tokens requires more compute). We have to design a complete yet minimal DSL. Probably the best way to do it is to use an iterative method, growing the DSL when noticing that certain tasks cannot be solved without new primitives.

### Algorithm

While solution is not found:

1. Generate `n` python functions given input-output pairs. 
2. Do `r` steps of iterative function refinement
3. Test-time training. The model has generated `m` python functions that are not correct, but they run and hopefully they are in the right direction. We can treat those functions as new tasks and do test-time training on them (we know the inputs, outputs and code)

The model will be given all the task inputs and the available outputs, and will have to write python code that implements the task. By giving all the inputs we force the model to create python code to generalize to inputs that do not have output.

## Tricks

- Teach how to use the DSL. It is important to create examples of how to use each function in the DSL
- Upsampling as data augmentation
- I can remove words from the tokenizer of a model to simplify grid representation.
- I could teach the model to draw. Given some painting generate code to create the painting. That might help the model to learn the 2d structure of the grids.
- Focus on an end to end approach. On ARC24 I lost the focus and mostly worked on pre-training. I should always evaluate the end to end system, although it requires more compute is the right way to do it.