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

The plot below shows the valid outputs probability and the dsl usage of Qwen2.5-Coder with different
sizes. The size is really important (at least for base models). Bigger models generate valid
outputs more frequently and use the dsl more frequently as well.

![alt text](res/1753292348173_image.png)

To be able to use the 14B model I had to use `tensor_parallel_size=2` and halve the `max_model_len=16000`.

I have also tried Qwen3 but it was slower and worse, so I will be using Qwen2.5-Coder-7B for the experiments
in this iteration.

## Conclusion

## Next steps

## TODO

- [x] How many primitive functions does BARC have? And Hodel?
- [x] Create a prompt with the available DSL functions
- [x] What is the best way to do inference?
  - [x] VLLM
  - [x] Pure python with caching
  - [ ] SGLang (I'm not going to try it this iteration)
- [x] What is the effect of the model size?
