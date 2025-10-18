# Iteration 34. Multi-turn RL

_18-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Implement a script to do multi-turn RL training, and test if it has a noticeable effect on model accuracy.

## Motivation

On [Iteration 28](Iteration_28_refine_predictions.md) I saw that the BARC induction model is not
good at refining its predictions. That forces us to just make independent predictions with the model.

But that is not efficient, we should take into account previous predictions to avoid
repeating errors and benefit from the execution feedback.

## Development

```bash
python scripts/multi-turn_rl_code_finetuning.py \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-18-debug-multi-turn-RL/baseline

[rank0]: NotImplementedError: Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead.
```

After changing from Dataset to IterableDataset I get this bad surprise.

## Results

## Conclusion

## Next steps

## TODO

- [ ] On a first step I have to modify the current RL script to train on a generator
- [ ] On a second step I have to couple the prompt generation and the training to be able to use the predictions done during GRPO training
