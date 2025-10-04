# Iteration 29. Multi-gpu RL

_04-10-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I train a model with RL and multiple GPUs?

## Motivation

The current unsloth implementation is able to train at around 3k-5k steps per day depending on the
number of generations. To be able to train faster I want to try to use multiple GPUs for training.

Current BARC induction model is not strong enough to solve ARC, it needs a lot of RL training.

## Development

### Documentation

<https://docs.unsloth.ai/basics/multi-gpu-training-with-unsloth>

- **DDP**: Distributed Data Parallel. The most common PyTorch distributed training method. Each GPU holds a full copy of the model, processes different batches of data, and gradients are synchronized across GPUs.
- **FSDP**. Fully Sharded Data Parallel. A PyTorch parallelism method where model parameters, gradients, and optimizer states are sharded across GPUs to save memory. Each GPU only holds a fraction of the model at any time.

I'm interested in DDP, because the model can fit on a single GPU.

### Implementation

```bash
# baseline
export EPOCHS=1
export NUM_GENERATIONS=8
export ACCUM_STEPS=2
python scripts/rl_code_finetuning.py \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.70 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-debug-multigpu/baseline-1GPU

# multigpu with torchrun
# this hangs after initializing the LLM engine
export EPOCHS=1
export NUM_GENERATIONS=8
export ACCUM_STEPS=2
cd scripts
torchrun --nproc_per_node 2 -m rl_code_finetuning_multigpu \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.70 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-debug-multigpu/2-GPUs

# accelerate
# this also hangs after initializing the LLM engine
accelerate launch rl_code_finetuning_multigpu.py \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.70 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-debug-multigpu/2-GPUs


# new script
export EPOCHS=1
export NUM_GENERATIONS=8
export ACCUM_STEPS=2
python scripts/rl_code_finetuning_multigpu.py \
--max-seq-length 1536 \
--max-completion-length 512 \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.3 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-debug-multigpu/new-script-1GPU
# 28/67 [16:41<20:54, 32.18s/it

# new script 2 gpus
accelerate launch scripts/rl_code_finetuning_multigpu.py \
--max-seq-length 1536 \
--max-completion-length 512 \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.3 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-debug-multigpu/new-script-2GPUs
# 27/33 [17:35<04:01, 40.32s/it
```

Baseline experiment trains at around 28s/it and the GPU is at 100% utilization.
It seems that unsloth is not currently prepared to do multigpu training, I will have to try with plain trl.

I had to modify the `GRPOTrainer` on line 539 to add `quantization='bitsandbytes',`, otherwise I don't have enough VRAM. `/home/gbarbadillo/miniconda3/envs/arc25/lib/python3.10/site-packages/trl/trainer/grpo_trainer.py`

When training with 2 GPUs the number of steps is halved, I would need to do experiments to verify
that the model is learning correctly.

### Validate implementation locally

```bash
# baseline
export EPOCHS=40
export NUM_GENERATIONS=8
export ACCUM_STEPS=2
python scripts/rl_code_finetuning.py \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--max-seq-length 1536 \
--max-completion-length 512 \
--gpu-memory-utilization 0.70 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small-10_arc-agi_training_challenges.json \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-validate-multigpu/baseline-1GPU-${EPOCHS}epochs

# new script with 1 GPU
export EPOCHS=40
export NUM_GENERATIONS=8
export ACCUM_STEPS=2
python scripts/rl_code_finetuning_multigpu.py \
--max-seq-length 1536 \
--max-completion-length 512 \
--learning-rate 1e-5 \
--epochs ${EPOCHS} \
--warmup-ratio 0.01 \
--gpu-memory-utilization 0.3 \
--num-generations ${NUM_GENERATIONS} \
--lora-r 16 \
--gradient-accumulation-steps ${ACCUM_STEPS} \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small-10_arc-agi_training_challenges.json \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-10-05-validate-multigpu/new-script-1GPU-${EPOCHS}epochs
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Validate the code locally
  - [x] Create an even smaller dataset ~ 10 tasks
  - [x] Train with the previous script on a single GPU
  - [ ] When training without unsloth it does not log the same metrics
  - [ ] Train with the new script on a single GPU
  - [ ] Train with the new script on 2 GPUs
