#!/usr/bin/env bash
source /root/arc25_env/bin/activate
source /root/secrets.sh
export PYTHONPATH=$PYTHONPATH:/root/arc25
export FOLDER=/root/trainings/2025-09-28-search-and-learn
export INITIAL_PREDICTIONS=16
export EPOCHS=32
export PREDICTIONS_PER_EPOCH=16
export GROUP_SIZE=20

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

learning_rates=(1e-5 5e-6 2e-6 1e-6)
for i in "${!learning_rates[@]}"; do
  export CUDA_VISIBLE_DEVICES=$i
  export LEARNING_RATE=${learning_rates[$i]}
  export EXPERIMENT_NAME=${INITIAL_PREDICTIONS}i_${EPOCHS}x${PREDICTIONS_PER_EPOCH}_lr${LEARNING_RATE}_${GROUP_SIZE}-group-size

  newpath=$(copy_to_tmp /root/arc25/scripts/search_and_learn_with_unsloth.py)
  python "$newpath" \
    --task-group-size ${GROUP_SIZE} \
    --initial-predictions ${INITIAL_PREDICTIONS} \
    --predictions-per-epoch ${PREDICTIONS_PER_EPOCH} \
    --learning-rate ${LEARNING_RATE} \
    --max-epochs ${EPOCHS} \
    --gpu_memory_utilization 0.75 \
    --model-path /data/uds-fourth-five-hunter-250929 \
    --dataset-path /root/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
    --output-dir ${FOLDER}/${EXPERIMENT_NAME} &
done

# wait for *all* background jobs to finish
echo "Waiting for all commands to finish..."
wait
echo "All commands finished"