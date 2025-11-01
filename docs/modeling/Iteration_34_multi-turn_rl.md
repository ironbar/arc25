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

All the evolutionary test-time compute methods are based on the capability of the model to use
feedback from execution.

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

### Cluster experiments

```bash
export BETA=0.02
export MAX_GRAD_NORM=0.05
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-19-multi-turn-rl
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/multi-turn_rl_code_finetuning.py \
--lora_r ${LORA_R} \
--beta ${BETA} \
--max-grad-norm ${MAX_GRAD_NORM} \
--no-load-in-4bit \
--reward-name ${REWARD_NAME} \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--save-steps 200 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/refine_dataset.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 245114.0

rsync -aPv -m  --include='*/'  --exclude *.pt --include='checkpoint-19699/***' --exclude='*'  calculon01:/mnt/scratch/users/gbarbadillo/arc25/trainings/2025-10-19-multi-turn-rl   /mnt/data/MEGA/TEMP/
```

## Results

If we compare the base model doing 128 independent predictions against doing 64 predictions and refining the best ones that do not solve the task with the fine-tuned model, we see a small improvement in the metrics when evaluating on ARC-AGI-1 evaluation set.

| initial predictions | refinement predictions | valid code | valid outputs | unique outputs | train_pixel_score | train_correct_grids | train_pass_rate | train_is_correct | test_pixel_score | test_correct_grids | test_pass_rate | test_is_correct | is_correct |
|---------------------|------------------------|------------|---------------|----------------|-------------------|---------------------|-----------------|------------------|------------------|--------------------|----------------|-----------------|------------|
| 128                 | 0                      | **99.9%**  | 71.7%         | **49.8%**      | 42.1%             | 2.4%                | 1.6%            | 16.3%            | 40.9%            | 2.0%               | 2.0%           | 23.0%           | 16.3%      |
| 64                  | 64                     | 93.7%      | **76.5%**     | 42.5%          | **48.7%**         | **3.1%**            | 1.6%            | **18.5%**        | **47.2%**        | **2.5%**           | **2.4%**       | **24.5%**       | **17.8%**  |

The improvement is more clear if we compare against the previous refinement experiment.

| fine-tuned model | valid code | valid outputs | unique outputs | train_pixel_score | train_correct_grids | train_pass_rate | train_is_correct | test_pixel_score | test_correct_grids | test_pass_rate | test_is_correct | is_correct |
|------------------|------------|---------------|----------------|-------------------|---------------------|-----------------|------------------|------------------|--------------------|----------------|-----------------|------------|
| no               | **99.7%**  | 74.0%         | **43.7%**      | 45.8%             | 2.1%                | 1.1%            | 16.5%            | 44.4%            | 1.7%               | 1.6%           | 21.5%           | 16.0%      |
| yes              | 93.7%      | **76.5%**     | 42.5%          | **48.7%**         | **3.1%**            | **1.6%**        | **18.5%**        | **47.2%**        | **2.5%**           | **2.4%**       | **24.5%**       | **17.8%**  |

The fine-tuning metrics look healthy, although we see at the end of the training that the model starts doing long predictions.

![alt text](res/1761973653869_image.png)

## Conclusion

We have observed a small improvement (16.3% -> 17.8% pass-rate) when doing prediction refinement with
a model fine-tuned with RL to do prediction refinement.
If we had an stable RL training and enough time and compute, maybe this small improvement could be make bigger.

## Next steps

- Is RL the best way to teach the model to refine its predictions? Maybe we should use supervised learning
  first, which has stronger learning signal.

## TODO

- [x] On a first step I have to modify the current RL script to train on a generator
- [x] Create two datasets of 10k tasks from BARC
- [x] Generate predictions to create a 2nd turn dataset for RL
- [x] Prepare the dataset for training
- [x] Train 2nd turn RL
- [x] Evaluate using the same setup from Iteration 28
