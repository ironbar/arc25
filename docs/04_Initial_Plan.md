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

Additionally we could define a continuous metric such as the number of correct pixels and use it with reinforcement learning to modify the model towards solutions
that score higher.

## Path 2. Human inspired approach

![](res/how-humans-solve-arc.png)

When humans try to solve ARC tasks we draw some hypothesis and test it in our heads, if it is not correct we update our beliefs and refine the hypothesis. What modules are needed to do this process?

- **Policy.** What action do I have to do to achieve the goal? Learned with hindsight
- **World model.** What happens if do this action? Learned with past experiences
- **Judgment.** Is the solution correct? Learned with human feedback or by comparison
- **Learning.** In difficult problems we are able to learn from our errors and modify our initial beliefs about the problem.

Reasoning is an iterative process, as shown in the loop diagram in the image.

### How o3 solved ARC-AGI-1?

My intuition is that o3 success in ARC-AGI-1 is likely due to an improved policy and better judgment. Vanilla LLMs are not good at judgment, but reasoning models need to learn to know if some answer is correct or wrong. By training with reinforcement learning the model improves its policy, it learns which strategies are good and which are bad to solve the ARC tasks. o3 very likely describes the task with natural language, generates the output grid and checks if the output looks correct. It might try different approaches, refine the description and when it is certain returns the response.

The main problems of o3 are:

- It generates 55k tokens per run, we probably cannot afford with the compute budget given in Kaggle.
- Does not seem to generalize well to more complex tasks with interacting rules such as ARC-AGI-2

### How AI might solve ARC?

We can reuse the diagram of how humans solve ARC and replace the elements.

![](res/how-ai-might-solve-arc.png)

Focusing on efficiency the best configuration for ARC might be the following:

- **Policy**: model, a Large Reasoning Model.
- **World model**: python interpreter
- **Judgment**: metric function
- **Learning**: reinforcement learning and hindsight experience replay

That way we only have to learn the policy and parametrize the learning, all the other modules are guaranteed to work perfectly.

## RL-ARC

The idea is to frame ARC as a reinforcement learning problem. The system is given a new task and it needs to learn it as efficiently as possible. It is like playing a game, but instead of hitting the buttons it has to write code to play.
The code generates an output that is evaluated against the ground truth and returns an score.

Finding the right program is equivalent to finding the right trajectory to solve a game. Instead of actions we write code, but the problem is exactly the same. When we want to solve a new task in ARC is the same as wanting to solve a new game. We can frame the problem as a Reinforcement learning game, with a very sparse reward.

The challenge is how to create a very efficient system to do this: how to design the DSL, how to pre-train the model, how to do split the compute between inference and test-time training... There is a huge number of possibilities that we need to explore
to find the winning system.

The DSL is perhaps one of the most critic parts of the system. Without a DSL the system will have to write
very long programs to solve the task, making the exploration of the solution space much harder and requiring
more compute (generating more tokens requires more compute). We have to design a complete yet minimal DSL. Probably the best way to do it is to use an iterative method, growing the DSL when noticing that certain tasks cannot be solved without new primitives.

### Algorithm

While solution is not found:

1. Generate `n` python functions given input-output pairs. `n=8` might be a good parametrization if we apply rotations and transpose to the original task.
2. Do `r` steps of iterative function refinement
3. Test-time training. The model has generated `m` python functions that are not correct, but they run and hopefully they are in the right direction. We can treat those functions as new tasks and do test-time training on them (we know the inputs, outputs and code). We could also apply RL techniques such as [GRPO](https://arxiv.org/abs/2402.03300).

The model will be given all the task inputs and the available outputs, and will have to write python code that implements the task. By giving all the inputs we force the model to create python code to generalize to inputs that do not have output.

### Model fine-tuning

The base model already knows how to code. In the fine-tuning phase we want to teach the model:

- How to use the primitive functions from the DSL (and how many primitive functions are there)
- The prior knowledge needed to solve ARC, for example to understand the 2d nature of the grids
- The intuition needed to solve ARC tasks using code
- The ability to reason: given the output to the code refine the code to reach the correct solution

All this knowledge will make learning at test time faster. Maybe we could solve all tasks from zero given enough compute and time, but ARC is all about efficiency and this fine-tuning will give the model the building blocks to solve the new tasks faster.

#### How to learn to reason from multi-turn conversations?

![alt text](res/1744637862310_image.png)

Imagine that we have a 3 turn conversation that ends with a correct answer. How can we use that data to teach the model to reason? We want to achieve two objectives: the model should create the correct function to solve the ARC task and if it doesn't it should be able to refine the function iteratively until it is correct.

We can create variations of the conversation of 2 and 1 turns.

TODO: I need to investigate this matter

One option would be to train and all the conversation variants. That could probably work. Deepseek trains on all the chain of thought if the final response is correct.

If we only trained the model in the latest response I'm not sure if it will need the reasoning, because the prefix for all the conversations would be the same and it could ignore the intermediate conversation.

On an online RL setup we would reward all CoT that generates the correct answer. We could shape the reward to give higher score to shorter answers, favouring efficiency. On an offline setup maybe curriculum learning could help by training first on the longer conversations and finally on the shorter ones.

## Tricks

- Teach how to use the DSL. It is important to create examples of how to use each function in the DSL. Also learning to combine the primitive functions might be important for ARC-AGI-2 because there is an emphasis on compositionality.
- Upsampling as data augmentation
- I can remove words from the tokenizer of a model to simplify grid representation.
- I could teach the model to draw. Given some painting generate code to create the painting. That might help the model to learn the 2d structure of the grids.
- Focus on an end to end approach. On ARC24 I lost the focus and mostly worked on pre-training. I should always evaluate the end to end system, although it requires more compute is the right way to do it.
- Being able to create embeddings with 2d information could be a boost for the model. That needs to be done when pre-training the model.
- Deepseek-R1 paper describes that for small LLMs it is better to use distillation from bigger models than to use RL.