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

## Results

## Conclusion

## Next steps

- How can we effectively explore the search space
  - Funsearch
  - Alphacode (Google DeepMind’s AlphaCode shows that this simple pipeline yields >90 % unique clusters even with off‑the‑shelf Transformer samplers)
- Another source of variability is using multiple LLMs

## TODO

- [ ]
