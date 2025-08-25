# Iteration 22. Test-time Training with BARC induction model

_25-08-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I improve the results on ARC-AGI-1 evaluation with the BARC induction model using test-time training?

## Motivation

I have the intuition that we need to combine search and learn to be able to solve novel tasks. Using [toy tasks](Iteration_08_improve_HER.md) I probed that a model was able to generalize outside its training distribution by training on hindsight relabeled wrong attempts to solve the task. I need to probe that the same technique is helpful for ARC.

## Development

My initial idea is to take the predictions from the previous iteration and fine-tune the BARC model on those using hindsight relabel. Then I will do inference again and hopefully
I will see improvements.

I believe I should see improvements with just one epoch (train + inference) but that
doing multiple epochs would yield the best results. I'm not going to worry about efficiency on this iteration, I just want to see if the technique works or it doesn't.

### Data generation

The first step is generate the data for training. The fastest way could be to generate
the data directly with the chat template from the BARC model.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
