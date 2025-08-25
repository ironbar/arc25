# Iteration 22. Test-time Training with BARC induction model

_25-08-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Can I improve the results on ARC-AGI-1 evaluation with the BARC induction model using test-time training?

## Motivation

I have the intuition that we need to combine search and learn to be able to solve novel tasks. Using [toy tasks](Iteration_08_improve_HER.md) I probed that a model was able to generalize outside its training distribution by training on hindsight relabeled wrong attempts to solve the task. I need to probe that the same technique is helpful for ARC.

## Development

My initial idea is to take the predictions from the previous iteration and fine-tune the BARC model on those using hindsight relabel. Then I will do inference again and hopefully
I will see improvements.

I believe I should see improvements with just one epoch (train + inference) but that
doing multiple epochs would yield the best results. I'm not going to worry about efficiency on this iteration, I just want to see if the technique works or it doesn't.

### Data generation

The first step is generate the data for training. The fastest way could be to generate
the data directly with the chat template from the BARC model.

### First trainings

```bash
# better work with a single gpu for debugging
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=10
export MAXSEQLEN=4096
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-25-hr-trainings/3090-GPUS${N_GPUS}-BARC-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora \
--no-resume_from_checkpoint

export N_GPUS=2
export STEPS=1000
export MAXSEQLEN=8192
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-25-hr-trainings/3090-GPUS${N_GPUS}-BARC-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--use-dora \
--use-rslora \
--no-resume_from_checkpoint
```

I had to solve a bug of my implementation when using gradient checkpointing, and modify the tokenizer
from Llama to add the pad token.

- Around 3s per instance when training with batch size 32 and 4096 max sequence length.
- That reduces to 1.6 seconds when using 2 GPUS, so scaling is nice because GPU usage is almost 100% all the time.
- If I increase the max_seq_len to 8192 the training time per sample increases to 2 seconds, but the memory
  seems to increase just from 13GB to 15GB so there might be room for bigger training sequences.
- Training on 3200 samples would take around 1h40min on my 2x3090 setup. I had to use 4 bit quantization,
  liger kernels and gradient checkpoint to avoid the OOM errors.

#### Data collator

The data collator adds a new labels field to the batch that allows to skip the user text.

```python
print(tokenizer(text))
{'input_ids': tensor([[128000, 128000, 128006,  ...,    198,  74694, 128009]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]])}
data_collator([tokenizer(text)])
{'input_ids': tensor([[128000, 128000, 128006,  ...,    198,  74694, 128009]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]]), 'labels': tensor([[ -100,  -100,  -100,  ...,   198, 74694,  -100]])}
```

In this case it is ignoring the end of text token because it is the same as the padding token.

## Results

## Conclusion

## Next steps

## TODO

- [ ] Prepare the training data.
  - [x] Small toy dataset
  - [ ] With and without data augmentation
  - [ ] With and without solved tasks
- [ ] Which LoRA parameters are compatible with VLLM?
- [ ] Train the model on the cluster
- [ ] Script for inference
