# State of the art

<!--- --->

I'm going to recap all the learnings from the previous [ARC24 challenge](https://www.kaggle.com/competitions/arc-prize-2024).

## [OpenAI solved the ARC challenge with a tuned version of o3](https://arcprize.org/blog/oai-o3-pub-breakthrough)

![o3 performance](res/2025-03-18-13-59-03.png)

It is not public, but it is very likely that o3 is trained just with reinforcement learning like r1. When [o1 was announced](https://openai.com/index/learning-to-reason-with-llms/) they said:

> Our large-scale reinforcement learning algorithm teaches the model how to think productively using its chain of thought in a highly data-efficient training process.

This sounds very similar to the training process described in the [r1 paper](https://arxiv.org/abs/2501.12948).
The only different thing is that in the table with the results there is a `samples` field that is 6 for low compute and 1024 for high compute. If we compare the numbers it seems that it is the number of times each task was tried to be solved. So there must be an aggregation mechanism to combine the responses from all the runs. It could be as simple as a voting mechanism and as complex as the model receiving as input the responses and choosing.

On average it uses 55k tokens per run. For 100 tasks that would be 5.5M output tokens if each task is run only once. If we are allowed 12 hours for the submission that would require an output speed of 127 token/second. That might be possible for a model like [Qwen2.5 1.5B](https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html), probably not for a model like Llama3 8B.

So in theory we could take a reasoning model such as [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) and fine-tune it to do ARC tasks using RL. We could reward the model for creating shorter and correct answers. OpenAI also offers a [service](https://openai.com/form/rft-research-program/) to do "Reinforcement Fine-tuning". 

Notice that OpenAI has decided [not to release o3](https://techcrunch.com/2025/02/12/openai-cancels-its-o3-ai-model-in-favor-of-a-unified-next-gen-release/), so we don't know how necessary the tuning for ARC was needed.

Mikel Bobel-Irizar did an [awesome analysis](https://anokas.substack.com/p/llms-struggle-with-perception-not-reasoning-arcagi) of the effect of task length on the accuracy of o3. We could
use upscaling as data augmentation so the model learns to work with bigger images. There is also another [blogpost](https://anokas.substack.com/p/o3-and-arc-agi-the-unsolved-tasks) with the unsolved evaluation tasks.

### Other reasoning models

![other reasoning model results](res/2025-03-18-14-48-47.png)

Other reasoning models such as r1 and Sonnet 3.7 but none of them achieve as high results as OpenAI's model. That does not happen in other fields such as mathematics, so probably OpenAI is using some 2d data for its RL training.

Interestingly in the [results](https://arcprize.org/blog/r1-zero-r1-results-analysis) we can see that r1 uses 6-11k tokens to solve each task. That is between 5 and 10 times less than o3.

## Test-time training (TTT)

Test-time training was arguably the biggest discovery of ARC24 challenge. In retrospective it is clear that if intelligence is all about adaptation to novelty, then we should not use frozen models but let them adapt to do new tasks. The MindsAI team found this approach but they decided not to make their solution public.

Probably the best implementation and description was done by [the Architects](https://arxiv.org/abs/2411.07279). There is also a paper named [The Surprising Effectiveness of Test-Time Training for Abstract Reasoning](https://arxiv.org/abs/2411.07279) and my own [solution](https://ironbar.github.io/arc24/05_Solution_Summary/) also used TTT.


## Code generation

Different attempts have tried using LLMs to generate python code to solve the ARC tasks. This induction approach has the advantage that the functions can be verified, whereas output grids from the transduction approach cannot be verified.
This allows to generate thousands of candidate solutions and filter all those that do not generate correct outputs
for the training samples. The main differences between this methods is how the model is prompted to generate the responses.

- [Summary of the progress in the public leaderboard in 2024](https://arcprize.org/blog/2024-progress-arc-agi-pub)
- [Jeremy Berman](https://jeremyberman.substack.com/p/how-i-got-a-record-536-on-arc-agi) uses an approach similar to [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)
- [Ryan Greenblatt](https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt) was the first to show that this approach could work and how it scaled with the number of predictions.

## Transduction and induction

https://arxiv.org/abs/2411.02272

## Other

[CodeIt: Self-Improving Language Models with Prioritized Hindsight Replay](https://arxiv.org/abs/2402.04858)