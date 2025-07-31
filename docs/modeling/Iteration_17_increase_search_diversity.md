# Iteration 17. Increase search diversity

_30-07-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I increase search diversity by doing variations to the prompt?

## Motivation

On the [previous iteration](Iteration_16_search_with_base_models.md) I tried to increase diversity by feeding previously generated functions to the model in the prompt and asking for new approaches. The effect was the opposite and diversity was reduced. LLMs (at least small LLMs) struggle with negation.

In this issue I will take a different approach and instead I will add variation to the prompt:

- Add hints about which DSL functions to use
- Shuffle the order of the training samples
- Prompt variations (I could prepare different prompts with LLMs)
- Data augmentation
- Temperature or other sampling parameters

The goal is to maximize diversity, but at the same time measure generation speed because generating multiple predictions with the same prompt is more
efficient than making multiple predictions with different prompts.

## Development

The work is done on the [notebook 009_search_with_base_models](../../notebooks/009_search_with_base_models.ipynb)

## Results

I have done 8 predictions for each of the 400 training tasks from ARC-AGI-1.
The metric of interest is the number of unique outputs, that is the best way to measure diversity.

| experiment                                       | valid code | valid outputs | unique outputs | dsl usage | pixel similarity | correct grids | solved task | inference time (s) | unique ratio |
|--------------------------------------------------|------------|---------------|----------------|-----------|------------------|---------------|-------------|--------------------|--------------|
| baseline                                         | 99.7%      | 78.6%         | 50.8%          | 56.3%     | 53.0%            | 1.9%          | 3.0%        | 667                | 64.64%       |
| shuffle train samples                            | 98.5%      | 74.1%         | 48.4%          | 56.8%     | 53.2%            | 2.0%          | 3.3%        | 1591               | 65.29%       |
| prompt variations (8)                            | 98.8%      | 73.7%         | 55.4%          | 41.1%     | 51.5%            | 1.9%          | 3.0%        | 2162.0             | 75.19%       |
| dsl suggestions                                  | 98.9%      | 71.7%         | 40.8%          | 70.3%     | 54.0%            | 1.5%          | 2.3%        | 1706               | 56.91%       |
| data augmentation                                | 98.2%      | 71.5%         | 46.3%          | 57.3%     | 52.4%            | 1.8%          | 3.3%        | 1699               | 64.80%       |
| data augmentation + shuffle train samples        | 98.2%      | 72.0%         | 45.4%          | 57.8%     | 52.9%            | 1.7%          | 3.0%        | 1674               | 63.09%       |
| shuffle train samples + remove last train sample | 98.5%      | 76.4%         | 46.8%          | 61.3%     | 54.1%            | 1.8%          | 3.0%        | 1336               | 61.33%       |
| 2 functions per prediction                       | 99.1%      | 68.5%         | 46.2%          | 55.9%     | 52.2%            | 1.7%          | 3.3%        | 1058.0             | 67.44%       |
| 4 functions per prediction                       | 98.0%      | 66.4%         | 45.9%          | 57.8%     | 50.1%            | 1.2%          | 1.5%        | 683                | 69.11%       |
| 8 functions per prediction                       | 97.6%      | 73.5%         | 33.8%          | 36.5%     | 46.4%            | 0.8%          | 0.5%        | 535.0              | 45.92%       |

- None of the experiments yielded an improvement in output diversity. When using prompt variations we
  can see an increase in 5% the unique outputs ratio, but at the cost of not using the dsl 15% less.
- Doing multiple inferences per prompt is faster and has more variability than the other techniques tried
- I find weird that data augmentation of shuffling the train samples does not increase the diversity.

I have to take in mind that the model can write different functions that generate the same output.
Thus the relation between the metric and the generated code is complex and requires understanding
of the effect of the code in the input data.

## Conclusion

Surprisingly none of the tried techniques were able to increase the diversity of the outputs. I'm afraid
I don't have yet a method to exhaustively search the solution space.

## Next steps

- How can we effectively explore the search space
  - Funsearch
  - Alphacode (Google DeepMind’s AlphaCode shows that this simple pipeline yields >90 % unique clusters even with off‑the‑shelf Transformer samplers)
- Another source of variability is using multiple LLMs
