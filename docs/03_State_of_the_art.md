# State of the art

<!--- --->

I'm going to recap all the learnings from the previous [ARC24 challenge](https://www.kaggle.com/competitions/arc-prize-2024).

## Summary of all the learnings

- Reasoning models trained with RL like `o3` can solve ARC-AGI-1, but they need a lot of compute. That is around [x40000 times](https://x.com/guille_bar/status/1870479630383329472) the compute allowed in the competition ($8 vs $340k). However on ARC-AGI-2 they seem to be scoring below 5%.
- Test-time training is crucial to improve the accuracy of transduction models. In my case the score improves from 11 to 33.
- Frontier LLMs can generate code that solves more than half of the semi-private ARC set.
- Induction and transduction are complementary approaches. It would have sense to first try with induction (which has higher guarantees) and use transduction only if induction fails.
- LLMs struggle with tasks that have big grids, however the fact that `o3` can solve ARC might hint that a 2d representation for the grid is not needed.

## [OpenAI solved the ARC challenge with a tuned version of `o3`](https://arcprize.org/blog/oai-o3-pub-breakthrough)

![o3 performance](res/2025-03-18-13-59-03.png)

Details are not public, but it is very likely that `o3` is trained just with reinforcement learning like `r1`. When [o1 was announced](https://openai.com/index/learning-to-reason-with-llms/) they said:

> Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process.

This sounds very similar to the training process described in the [`r1` paper](https://arxiv.org/abs/2501.12948).
The only different thing is that in the table with the results there is a `samples` field that is 6 for low compute and 1024 for high compute. If we compare the numbers it seems that it is the number of times each task was tried to be solved. So there must be an aggregation mechanism to combine the responses from all the runs. It could be as simple as a voting mechanism and as complex as the model receiving as input the responses and choosing.

On average it uses 55k tokens per run. For 100 tasks that would be 5.5M output tokens if each task is run only once. If we are allowed 12 hours for the submission that would require an output speed of 127 token/second. That might be possible for a model like [Qwen2.5 1.5B](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html), probably not for a model like Llama3 8B.

So in theory we could take a reasoning model such as [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) and fine-tune it to do ARC tasks using RL. We could reward the model for creating shorter and correct answers. OpenAI also offers a [service](https://openai.com/form/rft-research-program/) to do "Reinforcement Fine-tuning". 

Notice that OpenAI has decided [not to release `o3`](https://techcrunch.com/2025/02/12/openai-cancels-its-o3-ai-model-in-favor-of-a-unified-next-gen-release/), so we don't know how necessary the tuning for ARC was needed.

Mikel Bobel-Irizar did an [awesome analysis](https://anokas.substack.com/p/llms-struggle-with-perception-not-reasoning-arcagi) of the effect of task length on the accuracy of `o3`. We could
use upscaling as data augmentation so the model learns to work with bigger images. There is also another [blogpost](https://anokas.substack.com/p/o3-and-arc-agi-the-unsolved-tasks) with the unsolved evaluation tasks.

### Other reasoning models

![other reasoning model results](res/2025-03-18-14-48-47.png)

Other reasoning models such as `r1` and Sonnet 3.7 but none of them achieve as high results as OpenAI's model. That does not happen in other fields such as mathematics, so probably OpenAI is using some 2d data for its RL training.

Interestingly in the [results](https://arcprize.org/blog/r1-zero-r1-results-analysis) we can see that `r1` uses 6-11k tokens to solve each task. That is between 5 and 10 times less than `o3`.

## Test-time training (TTT)

Test-time training was arguably the biggest discovery of ARC24 challenge. In retrospective it is clear that if intelligence is all about adaptation to novelty, then we should not keep the models frozen but let them adapt to do new tasks. The MindsAI team found this approach but they decided not to make their solution public.

Probably the best implementation and description was done by [the Architects](https://arxiv.org/abs/2411.07279). There is also a paper named [The Surprising Effectiveness of Test-Time Training for Abstract Reasoning](https://arxiv.org/abs/2411.07279) and my own [solution](https://ironbar.github.io/arc24/05_Solution_Summary/) also used TTT.

Update: The MindsAI team has published [a paper](https://github.com/MohamedOsman1998/deep-learning-for-arc/blob/main/deep_learning_for_arc.pdf) describing their approach.

## Code generation

Different attempts have tried using LLMs to generate python code to solve the ARC tasks. This induction approach has the advantage that the functions can be verified, whereas output grids from the transduction approach cannot be verified.
This allows to generate thousands of candidate solutions and filter all those that do not generate correct outputs
for the training samples. The main differences between this methods is how the model is prompted to generate the responses.

- [Summary of the progress in the public leaderboard in 2024](https://arcprize.org/blog/2024-progress-arc-agi-pub)
- [Jeremy Berman](https://jeremyberman.substack.com/p/how-i-got-a-record-536-on-arc-agi) uses an approach similar to [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- [Ryan Greenblatt](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt) was the first to show that this approach could work and how it scaled with the number of predictions.

## [Transduction and induction](https://arxiv.org/abs/2411.02272)

This paper defined the terms transduction (generating the output grid directly) and induction (writing code to solve the tasks) and showed they were complimentary. Additionally they generated 400k new tasks using LLMs, showing that is possible to augment the data.

The [code](https://github.com/xu3kev/BARC) is open-source and I should take a look at it, it could serve as inspiration for creating the DSL.

## [ARC-AGI without pretraining](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html)

This novel approach does not use any training data! Scores 4.17 on ARC-AGI-2.

> We propose that lossless information compression can serve as an effective framework for solving ARC-AGI puzzles. A more efficient (i.e., lower-bit) compression of a puzzle correlates with a more accurate solution.

I don't understand the method well but it seems to be trying to create a compressed representation of the task, that is used to generate the output for the test sample.

## Reasoning, code and RL

### [CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://arxiv.org/abs/2402.04858)

This is a very interesting paper that uses code and hindsight experience replay. They use [Hodel's DSL](https://github.com/michaelhodel/re-arc) as a start point but they apply mutation to augment the tasks.

This paper shows that it's possible to learn from the test set using hindsight replay. How can we improve it?

- Using a bigger and better model, they use a small 220M LLM, we could be using a 7B parameter model
- Fine-tune the model individually for each task
- Do the search first on the training tasks to generate more training data
- More data augmentation
- Use a more simple and minimal DSL

### [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)

In this paper they train the models to use effectively the feedback from code execution by using reinforcement learning. 

- They train on 13k problems, an order of magnitude higher than ARC. The model is updated 12k times, each update is done with a batch size of 256. 
- The model is given 3 attempts to solve the tasks. 
- Only the final response is considered to compute the reward
- Trained took 5700 A100 GPU hours (20*288), that is around $10k. If I can work an order of magnitude below I would be fine. 
- The 70B model roughly doubles the performance of the 8B model. 
- Their implementation of SFT does not match the results from RL (This contradicts the R1 paper)

### [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)

They show that the model can develop capabilities such as self-verification, reflection  just with RL, without the need of SFT.

One interesting finding is that the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models.

### [Improving Multi-Turn Tool Use with Reinforcement Learning](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning)

TODO:

### [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

TODO:

## Other

### [Searching Latent Program Spaces](https://arxiv.org/abs/2411.08706)

They use an autoencoder to learn the space of programs. At inference the encoder gives a good starting point, but the gradient is used to find a better task representation. The idea is interesting but the performance is very weak, will have to wait if they are able to make it work at [Ndea](https://ndea.com/).

### [A 2D nGPT Model for Arc Prize](https://github.com/jfpuget/ARC-AGI-Challenge-2024/blob/main/arc.pdf)

Interesting because it uses a 2d transformer, not 1d as most of the other solutions.