# Iteration 16. Search with base models

_17-06-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I solve training ARC tasks using a base model with access to a DSL?

## Motivation

The [SOAR paper](https://icml.cc/virtual/2025/poster/43499) has shown that it is possible to use Qwen-Coder models to solve ARC tasks without the need of fine-tuning (although fine-tuning improves the scores.)

On this iteration I will try to replicate that work but instead of using plain python I will give access to a DSL that hopefully will ease the search process.

I could explore how to generate new functions, refine existing ones or combine multiple functions.

## Development

### Public DSLs

- [BARC](https://github.com/xu3kev/BARC): 54 primitive functions, solve around 160 training tasks.
  Does not have typed hints.
- [RE-ARC](https://github.com/michaelhodel/re-arc): 160 primitive functions, verified to be complete
  for the ARC-AGI-1 train set. The problem of this DSL is that the verifiers that implement the training
  task are very hard to understand. Maybe I can refactor it using Codex but seems like a very difficult task.

Having a big number of primitive functions will result in a bigger prompt if we want to give the LLM
the footprint of all the functions. Also make the combinatorial problem harder. The DSL should be
as minimal as possible.

### VLLM vs transformers

| Model                       | VLLM | VLLM quantized | transformers | transformers quantized | speedup | speedup quantized |
|-----------------------------|------|----------------|--------------|------------------------|---------|-------------------|
| Qwen2.5-Coder-0.5B-Instruct | 1107 | 563            | 48           | 36                     | 23.1    | 15.6              |
| Qwen2.5-Coder-1.5B-Instruct | 426  | 282            | 42           | 29                     | 10.1    | 9.7               |
| Qwen2.5-Coder-3B-Instruct   | 452  | 180            | 32           | 24                     | 14.1    | 7.5               |
| Qwen2.5-Coder-7B-Instruct   | 200    | 100            | -            | 27                     | -       | 3.7               |

VLLM is much faster than transformer when doing sequential inference (batch size 1).

And I also have noticed that VLLM benefits from making multiple predictions for the same prompt. For
example for the the following models we can increase the throughput (tokens/s) x8.

| batch size | Qwen2.5-Coder-0.5B-Instruct | Qwen2.5-Coder-7B-Instruct |
|------------|-----------------------------|---------------------------|
| 1          | 1107                        | 100                       |
| 2          | 1571                        | 135                       |
| 4          | 3199                        | 248                       |
| 8          | 3837                        | 354                       |
| 16         | 5845                        | 609                       |
| 32         | 6094                        | 870                       |

So for the smaller model we could generate around 6k tokens per second. That would be 1e9 tokens
if we generate with 4 GPUs for 12 hours. As a reference OpenAI generated 5.7e9 tokens to solve the test
set from ARC-AGI-1.

After seeing this huge differences in speed I will be using only VLLM on this iteration.

I have also seen that `enable_prefix_caching` allows faster execution the second time we call the model.
If the number of output tokens is small we can notice the inference speed, for example increasing to 150 tokens/second from an initial 75. For longer generations the effect is not that big.

### Try the Qwen2.5-Coder-14B-Instruct model

Just loading the 14B model with VLLM takes 4 minutes, although is a [4-bit quantized GGUF version](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF).
I had to also download the tokenizer from the [normal version](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)

It is capable of generating text at 100 tokens/s.

## Results

### Model size matters

#### 10 tasks, 64 predictions per task

The plot below shows the valid outputs probability and the dsl usage of Qwen2.5-Coder with different
sizes. The size is really important (at least for base models). Bigger models generate valid
outputs more frequently and use the dsl more frequently as well. All predictions are independent.

![alt text](res/1753292348173_image.png)

To be able to use the 14B model I had to use `tensor_parallel_size=2` and halve the `max_model_len=16000`.

I have also tried Qwen3 but it was slower and worse, so I will be using Qwen2.5-Coder-7B for the experiments
in this iteration.

#### 400 training tasks, 8 predictions per task

| parameters                  | 0.5B  | 1.5B  | 3B    | 7B    | 14B   |
|-----------------------------|-------|-------|-------|-------|-------|
| valid code                  | 87.6% | 98.5% | 96.3% | 99.6% | 99.9% |
| valid outputs               | 15.6% | 45.6% | 47.6% | 78.0% | 85.3% |
| unique outputs              | 13.2% | 39.5% | 39.6% | 50.6% | 28.4% |
| dsl usage                   | 29.4% | 30.9% | 40.3% | 56.5% | 87.6% |
| mean pixel similarity score | 32.5% | 43.4% | 47.8% | 52.9% | 59.1% |
| correct grids               | 0.1%  | 0.2%  | 0.8%  | 1.7%  | 1.8%  |
| solved tasks                | 0.0%  | 0.0%  | 0.8%  | 2.5%  | 3.8%  |
| inference time              | 460   | 425   | 1008  | 744   | 5505  |

This table shows the same experiment but with a bigger number of tasks. The sweet spot for my hardware
seems to be the Qwen2.5-Coder-7B model.

It can be seen that using bigger models leads to better metrics.

### Effect of the number of predictions with independent search

The following table shows results for the `Qwen2.5-Coder-7B` model when changing the number of
predictions and doing independent search. No data augmentation was used (which would likely increase the diversity of the predictions).

| predictions                 | 2     | 4     | 8     | 16    | 32    | 64    | 128   | 256   | 512   |
|-----------------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| valid code                  | 99.1% | 99.1% | 99.6% | 99.7% | 99.8% | 99.7% | 99.8% | 99.8% | 99.8% |
| valid outputs               | 75.4% | 78.2% | 78.0% | 79.2% | 79.0% | 78.8% | 79.1% | 79.0% | 79.1% |
| unique outputs              | 65.9% | 59.8% | 50.6% | 42.1% | 35.7% | 30.8% | 26.9% | 23.6% | 20.8% |
| dsl usage                   | 61.0% | 56.9% | 56.5% | 57.6% | 58.7% | 58.6% | 58.3% | 58.4% | 58.4% |
| mean pixel similarity score | 50.4% | 53.5% | 52.9% | 53.1% | 53.5% | 53.1% | 53.2% | 53.1% | 53.2% |
| correct grids               | 2.1%  | 1.8%  | 1.7%  | 1.9%  | 2.0%  | 1.8%  | 1.8%  | 1.8%  | 1.8%  |
| solved tasks                | 2.0%  | 2.0%  | 2.5%  | 4.8%  | 6.0%  | 7.5%  | 7.3%  | 9.5%  | 11.0% |
| inference time (s)          | 266   | 395   | 744   | 1366  | 2802  | 5681  | 10553 | 22596 | 41579 |
| inference time (h)          | 0.1   | 0.1   | 0.2   | 0.4   | 0.8   | 1.6   | 2.9   | 6.3   | 11.5  |

- The inference time increases ~linearly with the number of predictions
- The ratio of unique outputs decreases with the number of predictions, being just 20% with 512 predictions. Thus we could get the same results with around 100 predictions if we can always generate novel outputs.
- The number of solved tasks increases log-linearly with the number of predictions

![alt text](res/1761318484651_image.png)

This uses only one GPU, and we were able to do 512 predictions for 400 tasks in 12 hours. With 4 GPUs and 240 tasks, we could be making close to 4096 predictions per task.
Or if we devote half of the time to training, we could do 2048 predictions per task. Sounds like a big enough number.

In the next experiments I have to answer the question: can I improve this results using a more advanced search?
For the same compute budget (number of predictions), can we improve the metrics and/or the unique outputs ratio?

### Can we increase the unique outputs by conditioning on already generated code?

I have the belief that if we provide the model the already generated code and ask to implement different
approaches to the problem, the unique outputs will increase.

However the results below show that my intuition was wrong. In fact I get the opposite effect, the model
does not understand the instruction and the diversity is reduced. Furthermore in many of the tasks
copied the sample code exactly.

| experiment                           | valid outputs | unique outputs | exact duplicates |
|--------------------------------------|---------------|----------------|------------------|
| independent search                   | 87.3%         | 77.1%          | 0.0%             |
| sequential search v1                 | 91.8%         | 66.4%          | 13.0%            |
| sequential search v2                 | 92.6%         | 69.0%          | 8.0%             |
| sequential search v3                 | 91.0%         | 69.9%          | 5.8%             |
| sequential search v4 (system prompt) | 89.5%         | 71.8%          | 0.8%             |
| v4 + repetition_penalty=1.05         | 88.1%         | 72.7%          | 0.5%             |
| v4 + repetition_penalty=1.1          | 81.8%         | 72.1%          | 0.0%             |
| v4 + repetition_penalty=1.2          | 63.3%         | 61.1%          | 0.0%             |
| v1 +  repetition_penalty=1.05        | 91.9%         | 71.5%          | 4.8%             |
| v1 +  repetition_penalty=1.1         | 87.5%         | 70.3%          | 1.3%             |
| refine prompt v1                     | 95.5%         | 63.9%          | 19.0%            |

The results show that none of the sequential search experiments gets a similar unique output ratio
to simply do independent search. I was able to tune the prompt in the different versions of the
sequential search, and I saw improvements but the problem persisted.

Thus this is not a good strategy for searching (at least with this base model). I believe RL could
solve this, and maybe bigger models do better.

I have to look for diversity elsewhere. Maybe more worrying is that I have tried a prompt with the goal
of refining the existing code, and the exact duplicates rate was the highest of all the experiments: 19%.
Thus it seems that if I want to refine code, I'm going to throw many attempts to the trash can.

- [Gemini 2.5 Pro explanation](https://g.co/gemini/share/fdc61d549f8a)
- [o3 explanation](https://chatgpt.com/share/6888f931-ee80-8012-99eb-355934748062)

## Conclusion

This experimentation successfully demonstrated that a base code-generation model using a DSL can solve a subset of ARC tasks. 
The best experiment solved 11.5% of the training tasks.
The two most critical factors for success were model size and prediction volume. 
Larger models consistently produced higher-quality solutions and solved more tasks. 

The most effective strategy was a simple brute-force approach, generating hundreds of independent solutions per task, as this method significantly outperformed more complex search techniques aimed at refinement or increasing diversity. These "smarter" search prompts were counterproductive, often reducing the variety of generated outputs. While effective, the brute-force method is inefficient, suffering from diminishing returns in solution uniqueness.

## Next steps

- Try to increase search diversity
- Refine solution

## TODO

- [x] How many primitive functions does BARC have? And Hodel?
- [x] Create a prompt with the available DSL functions
- [x] What is the best way to do inference?
  - [x] VLLM
  - [x] Pure python with caching
  - [ ] SGLang (I'm not going to try it this iteration)
- [x] What is the effect of the model size?
