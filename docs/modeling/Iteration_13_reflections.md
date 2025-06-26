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

## Motivation

I need a solid theoretical foundation before I continue working to solve ARC.

## Analysis of the current problems

Despite adding more sample training tasks it only solved one real ARC task that is very easy.

### Bad exploration

The solution space is not fully explored, seems to be limited to repeat what it saw on training. This
is a huge problem, because without exploration(trying new approaches) we cannot solve any novel task.

On a game the action space is small, but the search space of an LLM is huge.

### Bias on the training data

There might be problems (bias) with the training data. If each primitive function is used just on a different
data distribution, the model might learn the association between code and inputs, ignoring the outputs completely.
Ideally the same input would be reused on all tasks.

If this problem exists, it's difficult for the model to develop the intuition needed to solve new tasks.

### Compositionality

Could a model trained on single step tasks learn to combine the steps to create multi-step tasks? I'm not sure.
Probably we should train for n steps, and hope that it could do novel combinations at test-time.

### Training data generation

Could an LLM generate novel tasks given code examples? Maybe because if we just give the information
as code, is a higher level abstraction than the ARC tasks and the domain of LLMs.

That was done in the transduction and induction paper, I need to revisit it. It would be a more direct
development because I would just write solutions for the ARC tasks and generators for the inputs.

And if later I see holes, I could create new tasks.

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
- Search won't be enough if the policy has wrong beliefs

### The bitter lesson

Search and learn, the two methods that can scale arbitrarily with compute, noticed brilliantly by [Richard Sutton](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

Search and learn are the two methods that allow to adapt to novelty. We can search a program to do a new task,
or we can learn the new task directly.

### Search in the program space

I like the search approach, but I believe it's better to search in the space of python programs. That allows
to execute the code and if the code is correct the output will be correct. This gives much more stronger
guarantees than learning or searching in the space of natural language programs.

However this requires to develop or to have a domain specific language (DSL). The expressivity of the
system will be tied to this DSL, if the DSL is not complete the system won't be able to solve ARC.

But I would argue that the same limitation applies to o3. Those transformations that weren't learned
on the training set, won't be able to be applied at test time. Because during training o3 has developed
its own natural language DSL.

If we want to search, we should probably train the model to do search. That would be done with RL.
Currently, I'm not doing that, just supervised fine-tuning on single turn conversation.

### Synthesis: Search and learn

My bet is to combine search and learning. I believe that it should be the most efficient approach and
that is how human intelligence works.

The system would search python programs that implement the new task. And it will refine its search
policy using hindsight experience replay. I believe that should reduce the number of search steps dramatically.

My current approach is failing at search (exploration). It does not explore the full solution space,
and repeats the errors over and over. This happens because:

- The model has not been trained to search, that requires RL or distillation from a model that knows
  how to search
- The current generation setup has no memory. All the predictions are independent, and that allows
  to repeat the errors over and over. I'm doing parallel search, when probably for ARC has more
  sense to do sequential search.

## Conclusion

## Next steps

- What is the max sequence length for training and inference on a GPU with 24GB of VRAM?
- Better sampling strategy. Could play with temperature, top_k and top_p to create more diverse samples. https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/text_generation#transformers.GenerationConfig.temperature
- What if I give hints of how to solve the problem int the prompt? Is the model capable on that case?
- What if I have a multi-turn conversation with the model to improve its own code?
- Reread transduction and induction paper, and code.
- Learn how to do RL with tool use
- Define the new data strategy. I might refactor an existing DSL. Given the typed hints to the model could allow to learn how to use it. I could try to generate tasks with LLMs.
- Better training objective. label_smoothing_factor might be used to preserve entropy. https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/trainer#transformers.TrainingArguments.label_smoothing_factor
- Validation might be solved ARC tasks. That way I could better measure the effect of the training tasks.
