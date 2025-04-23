# Iteration 4. First steps with code

_23-04-2025_

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
- One weird thing is that the 3B model has the Qwen-research license instead of Apache 2.0 license. The Qwen-research license does not allow for commercial use but I could probably use it for the ARC challenge.

### The right tool for each job

- [VLLM](https://blog.runpod.io/introduction-to-vllm-and-how-to-run-vllm-on-runpod-serverless/) seems to be the fasts option for inference. On my [ARC24 solution](https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24?scriptVersionId=199282752) I was able to make 96 predictions per task, and I believe the time used for inference was around 2 hours. Probably the architects didn't use VLLM because they created their own depth first. [Dynamically serving LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora.html#dynamically-serving-lora-adapters) 
- unsloth enables memory efficient and faster fine-tuning on a single gpu
- I could create some server to run the generated code on CPU

One interesting implementation would be to have independent services for: inference, fine-tuning and running code. And a master process would call this services, this master process would be very light because all the heavy work will be handed to the services.

### DSL definition

- There needs to be a document that describes the DSL, it might be a docstring in the python module.
- Polymorphism (functions that receive different data types) will make for a more simple DSL
- All primitive functions need tests
- All primitive functions need training samples of how to use them, we will use them to fine-tune the LLM.
- The training samples should have different levels of complexity. There should be very simple examples with just
one primitive function, and more complex examples with multiple primitive functions. I might need some criteria to validate training samples, for example I could test what would happen if removing lines of code and if the output does not change it means those lines are not necessary.
- LLMs could be helpful to generate new training samples (This was done in the [Transduction and induction](https://arxiv.org/abs/2411.02272) paper).
- I could use my own [DSL](https://github.com/ironbar/omni-arc/blob/main/omniarc/dsl.py) from ARC24 as a start point.

The idea is to define a first version of the DSL, train a first model on it and see how it performs on inference.

## Results

## Conclusion

## Next steps

- Base or Instruct model? On Qwen they recommend the base model if we are going to fine-tune.
- [lmdeploy](https://github.com/InternLM/lmdeploy) seems to be a faster alternative to VLLM

## TODO
