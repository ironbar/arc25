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

## Results

## Conclusion

## Next steps

## TODO

- [x] How many primitive functions does BARC have? And Hodel?
- [ ] Create a prompt with the available DSL functions
- [ ] What is the best way to do inference?
  - [ ] VLLM
  - [ ] SGLang
  - [ ] Pure python with caching
