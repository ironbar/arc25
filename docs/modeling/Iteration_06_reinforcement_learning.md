# Iteration 6. Reinforcement learning

_08-05-2025_

## Goal

Can we use RL to solve novel tasks?

## Motivation

I have already seen that Hindsight Experience Replay (HER) is helpful to adapt a model to new tasks. On
this iteration I want to try with RL, specifically with the popular Group Relative Policy Optimization (GRPO) algorithm.

Probably the best option is a combination of the two approaches, but first I have to find if GRPO is helpful.
I would also like to see how it compares with HER in terms of efficiency.

## Development

[TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) has already implemented GRPO, so testing it will be very easy.

I will be doing the experiments on the notebook [005_GRPO_exploration](../../notebooks/005_GRPO_exploration.ipynb)

### Reference GRPO parametrization

#### [Reinforcement Learning for Reasoning in Large Language Models with One Training Example](https://arxiv.org/abs/2504.20571)

- Qwen2.5-Math-1.5B
- 8 responses per prompt
- mini-batch size 128
- Batch size is 128 (number of prompts per step), they repeat the same prompt 128 times.
- learning rate: 1e-6
- temperature: 0.6
- Train for 2000 steps

#### [LLMs for Engineering: Teaching Models to Design High Powered Rockets](https://arxiv.org/abs/2504.19394)

- Batch size 64
- Qwen 2.5 7B

## Results

After solving the problem with timeouts I'm going to set the maximum number of tokens to 768, solving the task with 25 squares needed less than 600 tokens. That will enforce the model to keep the functions short.

[wandb](https://wandb.ai/guillermobarbadillo/20250508_GRPO_v2?nw=nwuserguillermobarbadillo)

9-vertical-lines:

- Solved on 61 steps with lr=1e-6 and num_generations=8
- With lr=2e-6 it is solved at step 21, 3m39s
- With lr=4e-6 it is solved in 48s, 5 steps

12-squares:

- with lr=4e-6 solves the task in 1194s, 73 steps.
- If I increase the number of generations to 16, it takes 150 steps and almost 100 minutes
- with lr=2e-5 and 16 generations it does not converge despite spending 140 minutes

16-squares:

- with lr=4e-6 the training diverges
- with lr=2e-6 I have stopped the training at epoch 123 and 70 minutes. It might seem that the training was going to diverge. At least I need something much faster.

## Conclusion

Despite doing a lot of experiments changing the learning rate and number of generations, I have not been able to solve a task with 16 squares when using GRPO. It solves the 9 vertical lines very fast, and it is capable of solving the 12 squares task but not consistently.

Either GRPO is less sample efficient than HER or I haven't found the right configuration. I believe it is the first hypothesis because GRPO only uses the information of the reward to learn, whereas HER can use the whole output so it uses much more information.

## Next steps

- I might revisit GRPO in the future. When reading papers that use GRPO, check the parametrization.
- Using VLLM would speedup the algorithm
