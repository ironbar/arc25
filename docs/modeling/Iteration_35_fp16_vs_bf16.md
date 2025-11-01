# Iteration 35. FP16 vs BF16

_01-11-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can we prevent RL training collapse if we use FP16 instead of BF16?

## Motivation

Yesterday I saw a lof of people talking in Twitter about using FP16 instead of BF16 for RL training.

![alt text](res/1761974415868_image.png)

They were refearing to the paper [Defeating the Training-Inference Mismatch via FP16](https://arxiv.org/abs/2510.26788).

> Reinforcement learning (RL) fine-tuning of large language models (LLMs) often suffers from instability due to the numerical mismatch between the training and inference policies. While prior work has attempted to mitigate this issue through algorithmic corrections or engineering alignments, we show that its root cause lies in the floating point precision itself. The widely adopted BF16, despite its large dynamic range, introduces large rounding errors that breaks the consistency between training and inference. In this work, we demonstrate that simply reverting to **FP16** effectively eliminates this mismatch. The change is simple, fully supported by modern frameworks with only a few lines of code change, and requires no modification to the model architecture or learning algorithm. Our results suggest that using FP16 uniformly yields more stable optimization, faster convergence, and stronger performance across diverse tasks, algorithms and frameworks. We hope these findings motivate a broader reconsideration of precision trade-offs in RL fine-tuning.

The challenge is almost over, but I haven't been able to find the root of the RL training collapse. So it
would be nice to see that doing this simple change solves the problem.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ]
