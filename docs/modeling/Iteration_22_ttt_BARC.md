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
--use-rslora
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
I have solved it by changing the pad token to `<|finetune_right_pad_id|>`.

### Training again in the cluster

#### First steps

I have updated the requirements of the environment, so the environment will have to be regenerated
in the cluster.

```bash
rsync -P /mnt/data/MEGA/TEMP/2025-08-25_evaluation-85640.json calculon01:/mnt/scratch/users/gbarbadillo/arc25/data

export N_GPUS=2
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=1000; condor_submit train.condor command="
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-25-hr-trainings/${N_GPUS}xA6000--${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-25-hr-trainings/${N_GPUS}xA6000--${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-no-dora \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

export N_GPUS=2
export LEARNING_RATE=5e-4
export MAXSEQLEN=8192
export STEPS=1000;
condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-25-hr-trainings/${N_GPUS}xA6000--${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-plain-lora \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--no-use-dora \
--no-use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

rsync -P -r calculon01:/mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-25-hr-trainings /mnt/data/MEGA/TEMP --exclude wandb/* --exclude *.pt
```

If I remove the gradient checkpointing I get OOM error when using the A6000 GPUs.

Training speed comparison (when using plain LoRA):

- 2xH100: 15.4s/it
- 2xA6000: 40.8s/it
- 2x3090: 46.7s/it

It is possible that we can speedup the H100 training because only 17% of the VRAM memory is being used
when using the same configuration as the other GPUs.

#### Speed tests

```bash
export N_GPUS=1
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=20;
export BATCH_SIZE=1
condor_submit train.condor command=" 
python \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-speed-tests/${N_GPUS}xA6000-${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-plain-lora-pdbs${BATCH_SIZE} \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size ${BATCH_SIZE} \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8


export N_GPUS=2
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=20;
condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-speed-tests/${N_GPUS}xA6000--${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-plain-lora \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

export N_GPUS=7
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=20;
condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-speed-tests/${N_GPUS}xA6000--${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-plain-lora \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 56 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r 32 \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=14 -append request_memory=80G
```

- 1 GPU: 71.63s/it
- 2 GPUS: 42.74s/it
- 4 GPUs: 26.82s/it
- 7 GPUs: Does not run successfully, probably OOM error but I'm not sure.

2 GPUs seems to be the sweet spot.

| n gpus | per-device-batch-size | batch time (s) | speedup | efficiency |
|--------|-----------------------|----------------|---------|------------|
| 1      | 1                     | 71.6           | 1       | 100.00%    |
| 2      | 1                     | 42.7           | 1.7     | 83.84%     |
| 4      | 1                     | 26.8           | 2.7     | 66.79%     |
| 7      | 1                     | -              | #VALUE! | #VALUE!    |
| 1      | 2                     | 75.2           | 1.0     | 95.21%     |
| 1      | 4                     | OOM            | #VALUE! | #VALUE!    |
| 1      | 8                     | OOM            | #VALUE! | #VALUE!    |

#### Lora rank sweep

```bash
export LORA_RANK=128
export N_GPUS=2
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=1000
condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-lora-rank/${N_GPUS}xA6000-${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-lora${LORA_RANK} \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r ${LORA_RANK} \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

rsync -P -r calculon01:/mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-lora-rank /mnt/data/MEGA/TEMP --exclude *.pt --include checkpoint-*000* --exclude checkpoint* --exclude wandb*
```

#### Influence of the number of training steps

```bash
export LORA_RANK=32
export N_GPUS=2
export LEARNING_RATE=1e-4
export MAXSEQLEN=8192
export STEPS=100
condor_submit train.condor command=" 
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning_hr.py \
--train_dataset_path /mnt/scratch/users/gbarbadillo/arc25/data/2025-08-25_evaluation-85640.json \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-27-training-steps/${N_GPUS}xA6000-${STEPS}steps-${MAXSEQLEN}msl-${LEARNING_RATE}lr-lora${LORA_RANK} \
--max-steps ${STEPS} \
--device-map None \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers ${N_GPUS} \
--logging-steps 1 \
--save-steps 100 \
--lora-r ${LORA_RANK} \
--no-use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8
```

### QLoRA is saving the whole model

It seems that when using QLoRA the whole quantized model is saved instead of just the adapter. I might
have to save the adapter manually to avoid moving large files.

```bash
export LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=1
export MAXSEQLEN=1024
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-26-qlora-issue/LoRA_${LORA_RANK} \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers 1 \
--lora-r ${LORA_RANK} \
--use-dora \
--use-rslora \
--no-use-4bit-quantization \
--no-resume_from_checkpoint

export LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=1
export MAXSEQLEN=1024
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-26-qlora-issue/qLoRA_${LORA_RANK} \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--max-seq-len ${MAXSEQLEN} \
--dataloader_num_workers 1 \
--logging-steps 1 \
--save-steps 1000 \
--lora-r ${LORA_RANK} \
--use-dora \
--use-rslora \
--use-4bit-quantization \
--no-resume_from_checkpoint
```

- The saved adapter weights 4.3GB if I use qLoRA, 2.2 if I use LoRA. The first result makes sense if
it is saving the whole 4bit quantized model. The second result does not make sense.
- Reducing the rank from 32 to 8 did not have any effect on the saved weight.
- Disabling the gradient checkpoint does not have any effect
- If I save the model manually the cause is clear, it is saving the embeddings layer because the size
  is changed when loading the model.

I have solved the issue by reusing the token `<|finetune_right_pad_id|>` that was already inside
the tokenizer instead of creating a new one.

### Check which LoRA versions are compatible with VLLM

I'm going to run very short train with the different configurations and see if they are compatible with VLLM.

```bash
export LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=1
export MAXSEQLEN=8192
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-26-lora-compatibility/qLoRA_${LORA_RANK}_dora_rslora \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers 1 \
--lora-r ${LORA_RANK} \
--use-dora \
--use-rslora \
--use-4bit-quantization \
--no-resume_from_checkpoint

export LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=1
export MAXSEQLEN=8192
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-26-lora-compatibility/qLoRA_${LORA_RANK}_rslora \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers 1 \
--lora-r ${LORA_RANK} \
--no-use-dora \
--use-rslora \
--use-4bit-quantization \
--no-resume_from_checkpoint

export LORA_RANK=8
export CUDA_VISIBLE_DEVICES=0
export N_GPUS=1
export STEPS=1
export MAXSEQLEN=8192
python scripts/finetuning_hr.py \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-08-26-lora-compatibility/qLoRA_${LORA_RANK} \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--batch-size 1 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 1 \
--save-steps 1000 \
--dataloader_num_workers 1 \
--lora-r ${LORA_RANK} \
--no-use-dora \
--no-use-rslora \
--use-4bit-quantization \
--no-resume_from_checkpoint
```

VLLM supports LoRA and RSLoRA, it does not support DoRA. Moreover I can give models on the fly, it seems that the first time is slower but otherwise speed looks to be the same.

```
sampling_params = SamplingParams(n=800, temperature=1.0, top_p=0.95, max_tokens=10)
Base model: 8000 tokens generated in 4.95 seconds (1614.81 tokens/second)
LoRA model: 8000 tokens generated in 5.78 seconds (1384.20 tokens/second)
RSLoRA model: 8000 tokens generated in 6.01 seconds (1330.54 tokens/second)
LoRA model: 8000 tokens generated in 5.26 seconds (1522.17 tokens/second)
RSLoRA model: 8000 tokens generated in 5.29 seconds (1512.97 tokens/second)
```

It seems that the first time a model is called it is slightly slower. And the LoRA model by itself is slightly slower than the base model. But manageable.

<details>
  <summary>ChatGPT summary of the 3 techniques</summary>

* **LoRA (Low-Rank Adaptation)**
  Freeze base weights $W$ and learn a low-rank update $\Delta W = \frac{\alpha}{r} BA$ with $A \in \mathbb{R}^{r\times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out}\times r}$. Cheap to train/serve, drop-in for Q/K/V/O and MLPs.

* **rsLoRA (rank-stabilized / root-scaled LoRA)**
  Same idea as LoRA, but changes the scaling (and init) so the update norm is \~invariant w\.r.t. rank (often $\alpha/\sqrt{r}$ instead of $\alpha/r$). More stable across different ranks; same runtime cost as LoRA.

* **DoRA (Weight-Decomposed LoRA)**
  Decomposes a weight into **direction** and **magnitude**; applies a low-rank update to the direction and learns a small per-channel magnitude (scale) too. Tends to boost quality vs plain LoRA, but needs explicit runtime support because of the decomposition step.

</details>

I believe then I should use rsLoRA and don't use DoRA for the following experiments.

### Inference script

```bash
python scripts/inference_with_BARC.py \
--base-model-path /home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-folder /mnt/hdd0/Kaggle/arc25/predictions/2025-08-27_first-finetuning-steps \
--lora-path /mnt/hdd0/MEGA/TEMP/2025-08-26-lora-rank/2xA6000--1000steps-8192msl-1e-4lr-lora32/checkpoint-1000


condor_submit train.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/inference_with_BARC.py \
--base-model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-folder /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-26-lora-rank/2xA6000--1000steps-8192msl-1e-4lr-lora32 \
--lora-path /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-08-26-lora-rank/2xA6000--1000steps-8192msl-1e-4lr-lora32/checkpoint-1000" -append request_gpus=1 -append request_cpus=4

rsync -P -r calculon01:/mnt/scratch/users/gbarbadillo/arc25/predictions /mnt/data/MEGA/TEMP
```

## Results

### LoRA rank

First trainings do not show any effect on the training metrics when changing the LoRA rank. Maybe I should
train for longer?

https://wandb.ai/guillermobarbadillo/2025-08-26-lora-rank/panel/wtf8lzs87?nw=nwuserguillermobarbadillo

### Training steps

Let's fix the lora rank to 32 and use different number of training steps.

TODO:

## Conclusion

## Next steps

- Do multiple iterations of search and learn

## TODO

- [ ] Prepare the training data.
  - [x] Small toy dataset
  - [x] With and without data augmentation
  - [ ] With and without solved tasks
- [x] Which LoRA parameters are compatible with VLLM? rsLoRA is compatible, DoRA isn't
- [x] Fix issue with qlora model saving the complete model
- [x] Train the model on the cluster
- [ ] Script for inference
  - [x] With support for LoRA
  - [ ] Add tests for data augmentation
  - [ ] Think if moving the prompt has sense
  - [ ] Including evaluation of the predictions, otherwise I have to do it on my computer.
  - [x] Try the script on the cluster
  - [x] There might be a problem with `os.environ['CUDA_VISIBLE_DEVICES'] = str(get_least_used_gpu_index())` on the cluster or on my computer. Probably it should only do changes if the variable is not set.
- [ ] Find best training hyperparameters (learning rate, batch size, lora rank, training steps)
- [x] Check training data: the order should be random
