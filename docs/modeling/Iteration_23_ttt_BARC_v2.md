# Iteration 23. All in with test-time training with BARC induction model

_02/09/2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Create an efficient implementation of test-time training with BARC that tries to solve
each task independently.

## Motivation

On the previous [iteration](Iteration_22_ttt_BARC.md) I have seen that TTT is able to improve
the solving rate of the BARC induction model. That experiment was done using all the
tasks at once. I already know from the previous competition that is better to solve each task independently,
I believe that creates a cleaner gradient signal.

Also I know from my [toy experiments](Iteration_08_improve_HER.md) that multiple iterations of search and learn are needed
to solve tasks that are far from the training distribution. Sometimes requiring in the
order of tens of epochs.

Thus in this iteration I want to implement an efficient way to do search and learn
in multiple epochs. If the implementation is successful it will very likely be part
of my solution for the 2025 challenge.

## Development

### Implementation ideas

- Inference with VLLM is very efficient, and I can use different LoRAs which is convenient for test-time training.
- trl could be used for training, although I don't know if it is the best option.
- I believe unsloth is integrated with VLLM, which will make inference as fast and maybe is the
  best way to do inference and training in the same process. Otherwise I would have to have a
  training service, an inference service and a master service that redirects the traffic between
  the two.

### Trying unsloth

[Documentation](https://docs.unsloth.ai/) is awesome.

### Training speed experiment

```bash
# baseline with huggingface and trl
export CUDA_VISIBLE_DEVICES=0
export LORA_RANK=32
export N_GPUS=1
export STEPS=100
export MAXSEQLEN=8192
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-03-speed-tests/LoRA${LORA_RANK}_${STEPS}steps_baseline-repeat \
--train-dataset-path /mnt/hdd0/Kaggle/arc25/data/hindsight_relabeled/2025-08-25_evaluation-no-data-augmentation-77.json \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--learning-rate 1e-5 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers ${N_GPUS} \
--lora-r ${LORA_RANK} \
--no-use-dora \
--use-rslora \
--use-4bit-quantization
```

```bash
# repeat with unsloth
export CUDA_VISIBLE_DEVICES=0
export LORA_RANK=32
export N_GPUS=1
export STEPS=100
export MAXSEQLEN=8192
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-03-speed-tests/LoRA${LORA_RANK}_${STEPS}steps_unsloth-remove-dropout \
--train-dataset-path /mnt/hdd0/Kaggle/arc25/data/hindsight_relabeled/2025-08-25_evaluation-no-data-augmentation-77.json \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--learning-rate 1e-5 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers ${N_GPUS} \
--lora-r ${LORA_RANK} \
--no-use-dora \
--use-rslora \
--use-4bit-quantization \
--use-unsloth
```

- The baseline trains 0.468 samples per second.
- First run with unsloth train 0.571 samples per second, slightly faster. However it uses just 36% memory instead of 64%
- When loading unsloth at the top of the script, speed improves to 0.615 samples per second
- Removing dropout from LoRA improves the speed to 0.629 samples per second
- Not using liger kernel seems to slow down to 0.618, but change is small3
- Using 8 bit quantization instead of 4 bit gets 0.605 samples per second
- Using an unquantized model improves the speed to 0.672 samples per second, 43% faster than the baseline.

So far we are seeing an speedup of 34% and a 50% reduction in VRAM usage when using unsloth. It might
be possible to trade that VRAM reduction for speed.

### Inference speed test

unsloth has a faster startup of 54s vs 1m51s for VLLM.

The table below shows the inference speed in tokens/s when generating 100 tokens per prompt.

| method \ n predictions | 8   | 32  | 128  | 512  |
|------------------------|-----|-----|------|------|
| VLLM                   | 140 | 512 | 1476 | 1992 |
| unsloth                | 138 | 510 | 1454 | 1464 |

They are very similar except from the last column, where I believe VLLM is using more VRAM memory than
unsloth. This is promising because it opens the door to use unsloth both for training and inference
in the same process.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Try unsloth for both training and inference
- [ ] Compare unsloth speed against trl and VLLM
