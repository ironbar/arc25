# Iteration 4. First steps with code

_start date_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

## Motivation


## Development

### Qwen2.5 Coder

[qwen2.5-coder-family](https://qwenlm.github.io/blog/qwen2.5-coder-family/) might be a good race-horse for ARC. There are many different model sizes: 0.5B, 1.5B, 3B, 7B, 14B and 32B. The smaller models have a context window of 32k tokens, from the 7B it is 128k.

I would start with the smallest model and there is always time to use a bigger model.

More information:

- Released on November 12, 2024
- Qwen2.5-Coder-32B-Instruct has become the current SOTA open-source code model, matching the coding capabilities of GPT-4o.
- Code repair is an important programming skill. Qwen2.5-Coder-32B-Instruct can help users fix errors in their code, making programming more efficient. 
- For each size, we open-sourced both Base and Instruct models, where the Instruct model serves as an official aligned model that can chat directly, and the Base model serves as a foundation for developers to fine-tune their own models.

![qwen models benchmark vs sizes](res/2025-04-09-12-46-56.png)

- Bigger models score higher in the benchmarks as expected, it resembles a typical log-linear relation.
- One weird thing is that the 3B model has the Qwen-research license instead of Apache 2.0 license.

## Results

## Conclusion

## Next steps

- Base or Instruct model? On Qwen they recommend the base model if we are going to fine-tune.

## TODO

- [ ] Is the model license important? https://gemini.google.com/app/c849d289d357a788