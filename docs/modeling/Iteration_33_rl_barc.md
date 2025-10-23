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
# 243910.0, collapsed

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

# Increase beta and decrease max grad norm, increase the number of generations
export BETA=0.02
export MAX_GRAD_NORM=0.05
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 245086.

# I want to see if I need to accumulate steps
export BETA=0.02
export MAX_GRAD_NORM=0.05
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=1
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 245087. CUDA error: an illegal memory access was encountered
# It is not an OOM, but very suspicious to see when I remove the accum steps
# the step the error happend the max completion length was 990, the greatest of all the steps
# However GPU memory usage was just 60%

export BETA=0.02
export MAX_GRAD_NORM=0.05
export REPETITION_PENALTY=1.02
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=2
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 245088. collapses

# increase beta and decrease max grad norm even more
export BETA=0.04
export MAX_GRAD_NORM=0.02
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 245089.

# Increase the number of generations to 64
export BETA=0.04
export MAX_GRAD_NORM=0.02
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=64
export ACUM_STEPS=8
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 246723.0 

# increase the number of generations even more to 128
export BETA=0.04
export MAX_GRAD_NORM=0.02
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=128
export ACUM_STEPS=16
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
condor_submit train.condor command="
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/rl_code_finetuning.py \
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
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/barc/dataset_100k.json.gz \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${EXPERIMENT_NAME}" -append request_gpus=1 -append request_cpus=${N_CPUS} -append request_memory=128G --append 'requirements = (TARGET.Machine == "calculon21.das-nano.com")'
# 247093.0

# sync checkpoints
rsync -aPv -m  --include='*/'  --exclude *.pt --include='checkpoint-*5000/***'  --include='checkpoint-*0000/***' --exclude='*'  \
calculon01:/mnt/scratch/users/gbarbadillo/arc25/trainings/2025-10-14-rl-barc   /mnt/data/MEGA/TEMP/
```

```bash
export EXPERIMENT=2025-10-14-rl-barc/8lora_lr2e-6_arc-v2-no-pixel-score_epochs1_16gen_2accum-steps_repetition-penalty-1.02_masked-truncate_unquantized_beta0.01
export CHECKPOINT=10000; condor_submit train.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/inference_with_BARC.py \
--n-predictions 128 \
--base-model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--lora-path /mnt/scratch/users/gbarbadillo/arc25/trainings/${EXPERIMENT}/checkpoint-${CHECKPOINT} \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--use-data-augmentation \
--output-folder /mnt/scratch/users/gbarbadillo/arc25/predictions/${EXPERIMENT}/checkpoint-${CHECKPOINT}/evaluation" -append request_gpus=1 -append request_cpus=12 --append 'requirements = (TARGET.Machine == "calculon19.das-nano.com")' -append request_memory=32G

export EXPERIMENT=2025-10-14-rl-barc/1lora_lr4e-6_arc-v2-no-pixel-score_epochs1_16gen_2accum-steps_repetition-penalty-1.02_masked-truncate_unquantized_beta0.01
export CHECKPOINT=5000; condor_submit train.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/inference_with_BARC.py \
--n-predictions 128 \
--base-model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--lora-path /mnt/scratch/users/gbarbadillo/arc25/trainings/${EXPERIMENT}/checkpoint-${CHECKPOINT} \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--use-data-augmentation \
--output-folder /mnt/scratch/users/gbarbadillo/arc25/predictions/${EXPERIMENT}/checkpoint-${CHECKPOINT}/evaluation" -append request_gpus=1 -append request_cpus=12 --append 'requirements = (TARGET.Machine == "calculon19.das-nano.com")' -append request_memory=32G
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

### Local evaluation

```bash
python scripts/inference_with_BARC.py \
--n-predictions 8 \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--use-data-augmentation \
--output-folder /mnt/hdd0/Kaggle/arc25/predictions/2025-10-14-rl-barc/baseline/evaluation


export EXPERIMENT=2025-10-14-rl-barc/8lora_lr2e-6_arc-v2-no-pixel-score_epochs1_16gen_2accum-steps_repetition-penalty-1.02_masked-truncate_unquantized_beta0.01
for CHECKPOINT in 1000 5000 10000 15000 20000; do
  echo "Running inference for checkpoint-${CHECKPOINT}..."
  python scripts/inference_with_BARC.py \
    --n-predictions 8 \
    --lora-path /mnt/hdd0/MEGA/TEMP/${EXPERIMENT}/checkpoint-${CHECKPOINT} \
    --dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
    --use-data-augmentation \
    --output-folder /mnt/hdd0/Kaggle/arc25/predictions/${EXPERIMENT}/checkpoint-${CHECKPOINT}/evaluation
done

export EXPERIMENT=2025-10-14-rl-barc/1lora_lr4e-6_arc-v2-no-pixel-score_epochs1_16gen_2accum-steps_repetition-penalty-1.02_masked-truncate_unquantized_beta0.01
for CHECKPOINT in 1000 5000 10000 15000 20000; do
  echo "Running inference for checkpoint-${CHECKPOINT}..."
  python scripts/inference_with_BARC.py \
    --n-predictions 8 \
    --lora-path /mnt/hdd0/MEGA/TEMP/${EXPERIMENT}/checkpoint-${CHECKPOINT} \
    --dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
    --use-data-augmentation \
    --output-folder /mnt/hdd0/Kaggle/arc25/predictions/${EXPERIMENT}/checkpoint-${CHECKPOINT}/evaluation
done
```

### Train on StrongCompute workstation

I'm going to try to train directly in the workstation as a workaround because all the normal trainings
abruptly ended in 1 day. Maybe the workstation lasts more than one day. Another advantage is that I 
can use machines with just one GPU.

```bash
# arc25
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.04
export MAX_GRAD_NORM=0.02
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

# arc25_1
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.1
export MAX_GRAD_NORM=0.01
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

# arc25_2
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.2
export MAX_GRAD_NORM=0.005
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

# arc25_3
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.4
export MAX_GRAD_NORM=0.002
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

# arc25_1
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.04
export MAX_GRAD_NORM=0.01
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

#arc25_2
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.04
export MAX_GRAD_NORM=0.004
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}

#arc25_3
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export BETA=0.04
export MAX_GRAD_NORM=0.002
export REPETITION_PENALTY=1.01
export FOLDER=2025-10-14-rl-barc
export LEARNING_RATE=4e-6
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export N_CPUS=20
export LORA_R=1
export EPOCHS=1
export REWARD_NAME=arc-v2-no-pixel-score
export EXPERIMENT_NAME=${LORA_R}lora_lr${LEARNING_RATE}_${MAX_GRAD_NORM}max-grad-norm_${REWARD_NAME}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_repetition-penalty-${REPETITION_PENALTY}_masked-truncate_unquantized_beta${BETA}
python /root/arc25/scripts/rl_code_finetuning.py \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/data/barc/dataset_100k.json.gz \
--output-dir /root/trainings/${FOLDER}/${EXPERIMENT_NAME}
```

### Dataset from generator

When I increased the number of generations to 64 I had to increase the RAM memory in the cluster. Maybe
starting from a generator could reduce the memory requirements.

```bash
python scripts/rl_code_finetuning.py \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-23-debug-generator/debug

export NUM_GENERATIONS=32; python scripts/rl_code_finetuning.py \
--num-generations ${NUM_GENERATIONS} \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_100k.json.gz \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-23-debug-generator/debug-barc-${NUM_GENERATIONS}
```

This implementation starts with 13GB of RAM usage, grows to 17.2GB when loading the dataset, and just to 18.1GB
when creating the dataset for training. So apparently is very RAM memory efficient. Previous implementation
raised RAM usage to 26GB with the same configuration.

```bash
export NUM_GENERATIONS=32
export ACCUMULATION_STEPS=4

python scripts/rl_code_finetuning.py \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACCUMULATION_STEPS} \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_100k.json.gz \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-23-debug-generator/debug-barc_${NUM_GENERATIONS}generations_${ACCUMULATION_STEPS}accum-steps_from_list
# start RAM: 13.6GB, load dataset: 17.4GB, prepare dataset for training: 70GB (needed to use swap memory)

python scripts/rl_code_finetuning.py \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACCUMULATION_STEPS} \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems/dataset_100k.json.gz \
--epochs 1 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-23-debug-generator/debug-barc_${NUM_GENERATIONS}generations_${ACCUMULATION_STEPS}accum-steps_from_generator
# start RAM: 8GB, load dataset: 11.6GB, prepare dataset for training: 12.6GB
```

In the previous implementation I needed 70GB, with the new just 12GB. This explains the memory error that I saw in the cluster.

## Results

### Training still collapses and the model makes nonsense predictions

Despite training on a huge dataset, the training with lora rank 1 has collapsed.

![alt text](res/1760774465625_image.png)

I'm already using `beta=0.01, repetition_penalty=1.02, and max_grad_norm=0.1`, but I'm going to make
those constraints harder. Also I'm going to double the number of generations from 16 to 32.

RL is showing signs of improvements. The improvements are modest but noticeable.
The problem is that training for longer will likely make the improvements bigger, but I don't have a robust training configuration yet.
And the challenge ending is approaching.

## Conclusion

## Next steps

## TODO

- [x] Download and curate the synthetic datasets: https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems
- [ ] Could the random RAM problems be caused by evaluating the code generated by the LLM?
- [ ] Is the model improving when training on BARC data?
- [ ] Can I find a training configuration that allows me to train on the whole datasets without collapsing?
  I have launched multiple experiments with different configurations of kl loss and max_grad_norm to see
  if any works and which one break.