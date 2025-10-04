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

### Experiments

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
```

Baseline experiment trains at around 28s/it and the GPU is at 100% utilization.
It seems that unsloth is not currently prepared to do multigpu training, I will have to try with plain trl.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
