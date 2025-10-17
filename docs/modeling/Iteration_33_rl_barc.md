# Iteration 33. RL with BARC data

_14-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Train with RL from BARC synthetic datasets. Does it solve the collapsing problems? Does it produce stronger models?

## Motivation

Maybe the collapsing problems comes from repeating the 400 training tasks over and over. Since we
have the BARC synthetic datasets readily available, we should try to use them to see if we can
get stronger models and solve the training collapse problem.

## Development

The Heavy version of the [dataset](https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems)
is the most interesting one because it uses more seed functions and uses the stronger gpt4 model for description generation.

### Verify that I can train locally

```bash
python scripts/rl_code_finetuning.py \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_100k.json.gz \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-14-debug-BARC/debug
```

Works perfectly.

### Train and evaluate on cluster

```bash
export BETA=0.01
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=16
export ACUM_STEPS=2
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${REWARD_NAME}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
--beta ${BETA} \
--no-load-in-4bit \
--reward-name ${REWARD_NAME} \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 243910.0

export BETA=0.01
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=2e-6
export NUM_GENERATIONS=16
export ACUM_STEPS=2
export N_CPUS=20
export LORA_R=8
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${REWARD_NAME}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
--beta ${BETA} \
--no-load-in-4bit \
--reward-name ${REWARD_NAME} \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 243912.0
```

```bash
export STEPS=1000; export FOLDER=2025-08-29-smaller-datasets/2xA6000-${STEPS}steps-8192msl-1e-4lr-lora32; condor_submit train.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/inference_with_BARC.py \
--n-predictions 128 \
--base-model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--lora-path /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/checkpoint-${STEPS} \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--use-data-augmentation \
--output-folder /mnt/scratch/users/gbarbadillo/arc25/predictions/${FOLDER}/evaluation" -append request_gpus=1 -append request_cpus=20 --append 'requirements = (TARGET.Machine == "calculon19.das-nano.com")' -append request_memory=32G
```

### New memory limit when doing code execution

```bash
export LIMIT_MB=512; python scripts/rl_code_finetuning.py \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-15-debug-memory_limit/limit_${LIMIT_MB}MB \
--code-execution-memory-limit-mb ${LIMIT_MB}

export LIMIT_MB=256; python scripts/rl_code_finetuning.py \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-15-debug-memory_limit/limit_${LIMIT_MB}MB \
--code-execution-memory-limit-mb ${LIMIT_MB}
```

I cannot see the effect on the metrics... That's weird.

### Update inference to also evaluate and save output compressed

I'm going to update the inference script to also evaluate and save the results compressed. That way
I can quickly now the results of the inference.

```bash
python scripts/inference_with_BARC.py \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small_arc-agi_training_challenges.json \
--output-folder /mnt/hdd0/Kaggle/arc25/predictions/2025-09-15-debug-grpo/lr1e-5_small-dataset_80epochs_16gens_continue/small-training \
--lora-path /mnt/hdd0/Kaggle/arc25/trainings/2025-09-15-debug-grpo/lr1e-5_small-dataset_80epochs_16gens_continue/checkpoint-5360 \
--n-predictions 8
```

## Results

## Conclusion

## Next steps

## TODO

- [x] Download and curate the synthetic datasets: https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
- [ ] Could the random RAM problems be caused by evaluating the code generated by the LLM?