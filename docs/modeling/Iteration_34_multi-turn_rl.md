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

### Unsloth GRPO does not support Iterable datasets

```bash
python scripts/multi-turn_rl_code_finetuning.py \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-18-debug-multi-turn-RL/baseline

[rank0]: NotImplementedError: Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead.
```

After changing from Dataset to IterableDataset I get this bad surprise.

### Proof of concept with pre-generated responses

The easiest way to test the concept is to generate a dataset were I generate predictions
for the task and pick one that is not correct. This is exactly the same I did in [Iteration 28](Iteration_28_refine_predictions.md) but instead of doing it at test time, I need to do it
at training time using training data.

So the best option would be to take BARC dataset and make predictions for the tasks.

Making 8 predictions for 1000 tasks takes around one hour on a single GPU. A good proof of concept will require between 10k and 20k prompts, at least that is what I'm currently
training with RL before the training collapses.

#### Inference Dataset preparation

To prepare the dataset for inference I'm going to reuse the notebook `notebooks/016_prepare_BARC_data_for_training.ipynb`.

#### Inference

```bash
export PART=1
python scripts/inference_with_BARC.py \
--n-predictions 8 \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_10k_part${PART}.json.gz \
--use-data-augmentation \
--output-folder /mnt/hdd0/Kaggle/arc25/predictions/2025-10-18-barc-inference/part${PART}

export PART=2
python scripts/inference_with_BARC.py \
--n-predictions 8 \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_10k_part${PART}.json.gz \
--use-data-augmentation \
--output-folder /mnt/hdd0/Kaggle/arc25/predictions/2025-10-18-barc-inference/part${PART}
```

#### Dataset for 2nd turn conversation preparation

I have done the work on the already existing notebook `notebooks/014_refine_solutions.ipynb`.
The maximum prompt length is 8511, so I can keep the training parameters as they were.

## Results

## Conclusion

## Next steps

## TODO

- [x] On a first step I have to modify the current RL script to train on a generator
- [ ] Create two datasets of 10k tasks from BARC
- [ ] Generate predictions to create a 2nd turn dataset for RL
- [ ] Prepare the dataset for training
- [ ] Train 2nd turn RL
- [ ] Evaluate using the same setup from Iteration 28
