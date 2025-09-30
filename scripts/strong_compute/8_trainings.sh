#!/usr/bin/env bash
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export FOLDER=$OUTPUT_PATH/2025-09-28-search-and-learn

export FOLDER=$OUTPUT_PATH/2025-09-30-rl-collapse-study
export NUM_GENERATIONS=32
export ACUM_STEPS=4
export LEARNING_RATE=1e-6
export N_CPUS=20
export LORA_R=32
export EPOCHS=40

copy_to_tmp() {
    local filepath="$1"
    if [[ ! -f "$filepath" ]]; then
        echo "Error: file '$filepath' does not exist" >&2
        return 1
    fi

    # create temp dir (deleted automatically when system cleans /tmp)
    local tmpdir
    tmpdir=$(mktemp -d) || return 1

    # copy file to temp dir
    local filename
    filename=$(basename "$filepath")
    local dst="$tmpdir/$filename"

    cp "$filepath" "$dst" || return 1

    # print the new path
    echo "$dst"
}

# Experiment 1
export CUDA_VISIBLE_DEVICES=1
export REPETITION_PENALTY=1.00
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}_unmasked-truncated-completions
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--no-mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 2
export CUDA_VISIBLE_DEVICES=2
export REPETITION_PENALTY=1.01
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}_unmasked-truncated-completions
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--no-mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 3
export CUDA_VISIBLE_DEVICES=3
export REPETITION_PENALTY=1.02
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}_unmasked-truncated-completions
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--no-mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 4
export CUDA_VISIBLE_DEVICES=4
export REPETITION_PENALTY=1.05
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}_unmasked-truncated-completions
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
--num-generations ${NUM_GENERATIONS} \
--gradient-accumulation-steps ${ACUM_STEPS} \
--learning-rate ${LEARNING_RATE} \
--lora_r ${LORA_R} \
--repetition-penalty ${REPETITION_PENALTY} \
--epochs ${EPOCHS} \
--no-mask-truncated-completions \
--scale-rewards batch \
--gpu_memory_utilization 0.3 \
--warmup-ratio 0.01 \
--max-seq-length 9700 \
--max-completion-length 1024 \
--n-jobs ${N_CPUS} \
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 5
export CUDA_VISIBLE_DEVICES=5
export REPETITION_PENALTY=1.01
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 6
export CUDA_VISIBLE_DEVICES=6
export REPETITION_PENALTY=1.02
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 7
export CUDA_VISIBLE_DEVICES=7
export REPETITION_PENALTY=1.05
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# Experiment 8
export CUDA_VISIBLE_DEVICES=0
export REPETITION_PENALTY=1.10
export EXPERIMENT_NAME=lr${LEARNING_RATE}_epochs${EPOCHS}_${NUM_GENERATIONS}gen_${ACUM_STEPS}accum-steps_${LORA_R}lora_simplified-reward_repetition-penalty-${REPETITION_PENALTY}
newpath=$(copy_to_tmp /root/arc25/scripts/rl_code_finetuning.py)
python $newpath \
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
--model-path /data/uds-fourth-five-hunter-250929 \
--dataset-path /root/arc25/data/arc-prize-2024/arc-agi_training_challenges.json  \
--output-dir $FOLDER/$EXPERIMENT_NAME &

# wait for *all* background jobs to finish
echo "Waiting for all commands to finish..."
wait
echo "All commands finished"