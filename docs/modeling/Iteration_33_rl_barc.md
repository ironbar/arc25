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

### Train on cluster

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
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Download and curate the synthetic datasets: https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
