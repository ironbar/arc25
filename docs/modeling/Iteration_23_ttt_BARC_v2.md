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

I have verified that I can do fast inference and fast training in the same process with unsloth. Thus
I'm going to implement the algorithm with unsloth and unless I see performance problems I will stick
with it until the end of the challenge.

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

### Search and Learn algorithm

This is how one epoch of the search and learn algorithm would look like, the algorithm works on a single task:

1. Make n predictions with the model. Use data augmentation to increase the diversity of the predictions.
2. Parse the code from the predictions and execute the code to get the output grids
3. Evaluate the outputs on the training samples of the task
4. There could be some stopping criteria, for example if I have two different solutions that solve
   all the training tasks.
5. Prepare the data for training. I could sort them by the number of correct grids or other metrics.
   I could remove already predicted solutions on previous epochs. Using hindsight relabelling we
   generate new tasks for training.
6. Finetune the model
7. Repeat until the stop criteria is met or the number of maximum epochs is reached
8. Select the predictions for submission

Using a smaller number of predictions could be more efficient, according to [previous experiments with toy tasks](./Iteration_08_improve_HER.md#number-of-generations). Once the algorithm is implemented I will have
to tune all the hyperparameters. On Kaggle I will have 12 minutes per task when running the algorithm in parallel on the 4 GPUs.

#### Validate inference

I have done experiments on the training dataset to validate that the inference is correct and gives
the same results as in previous iterations.

```bash
# 400 training tasks, 20m49
n_preds	valid code	valid outputs	unique outputs	train_pixel_score	train_correct_grids	train_pass_rate	train_is_correct	test_pixel_score	test_correct_grids	test_pass_rate	test_is_correct	is_correct
MEAN	8.0	1.0	0.774	0.616	0.484	0.121	0.107	0.26	0.474	0.113	0.112	0.278	0.258
# this validates the inference pipeline, results are very similar as the shown below

# baseline, runtime around 21 minutes with batch size 8
	n_preds	valid code	valid outputs	unique outputs	pixel similarity	correct grids	train_pass_rate	train_pass@n	pass_rate	pass@n
MEAN	8.0	1.0	0.753125	0.6175	0.594594	0.129178	0.105435	0.2225	0.104452	0.2175
MEAN	8.0	1.0	0.758125	0.625313	0.602329	0.12629	0.103494	0.2625	0.101396	0.26
MEAN	8.0	1.0	0.765625	0.615938	0.611174	0.148103	0.12081	0.27	0.11822	0.265
MEAN	8.0	1.0	0.761875	0.614688	0.595542	0.130861	0.113375	0.2625	0.111104	0.2625
```

#### Effect of 4 bit quantization at inference

```bash
# training
# unquantized, 20m49
# same but with 4 bit quantization, 24m24
n_preds	valid code	valid outputs	unique outputs	train_pixel_score	train_correct_grids	train_pass_rate	train_is_correct	test_pixel_score	test_correct_grids	test_pass_rate	test_is_correct	is_correct
MEAN	8.0	1.0	0.774	0.616	0.484	0.121	0.107	0.26	0.474	0.113	0.112	0.278	0.258
MEAN	8.0	1.0	0.771	0.63	0.468	0.103	0.088	0.248	0.457	0.095	0.095	0.285	0.242

# evaluation
# unquantized 29m15
# 4bit quantization, 34m36
	n_preds	valid code	valid outputs	unique outputs	train_pixel_score	train_correct_grids	train_pass_rate	train_is_correct	test_pixel_score	test_correct_grids	test_pass_rate	test_is_correct	is_correct
MEAN	8.0	1.0	0.709	0.633	0.413	0.021	0.013	0.058	0.402	0.016	0.016	0.07	0.058
MEAN	8.0	1.0	0.708	0.634	0.415	0.022	0.015	0.058	0.404	0.018	0.018	0.068	0.058
```

It seems that quantization makes inference slower, but accuracy seems to be the same.

### Cluster experiments

#### First steps

```bash
export FOLDER=2025-09-07-search-and-learn
export N_PREDICTIONS=128; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions ${N_PREDICTIONS} \
--max-epochs 0 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/baseline_${N_PREDICTIONS}" -append request_gpus=1 -append request_cpus=32
```

```bash
export FOLDER=2025-09-07-search-and-learn
export N_PREDICTIONS=128
export LEARNING_RATE=1e-5; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions 64 \
--predictions-per-epoch 64 \
--learning-rate ${LEARNING_RATE} \
--max-epochs 1 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/2partitions_${N_PREDICTIONS}_lr${LEARNING_RATE}" -append request_gpus=1 -append request_cpus=32
```

```bash
export FOLDER=2025-09-07-search-and-learn
export N_PREDICTIONS=128
export LEARNING_RATE=1e-4; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions 32 \
--predictions-per-epoch 32 \
--learning-rate ${LEARNING_RATE} \
--max-epochs 3 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/4partitions_${N_PREDICTIONS}_lr${LEARNING_RATE}" -append request_gpus=1 -append request_cpus=32
```

```bash
export FOLDER=2025-09-07-search-and-learn
export N_PREDICTIONS=128
export LEARNING_RATE=1e-4; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions 16 \
--predictions-per-epoch 16 \
--learning-rate ${LEARNING_RATE} \
--max-epochs 7 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/8partitions_${N_PREDICTIONS}_lr${LEARNING_RATE}" -append request_gpus=1 -append request_cpus=32
```

#### Debugging

```bash
export FOLDER=2025-09-07-debug-search
export BATCH_SIZE=8; export N_PREDICTIONS=128; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions ${N_PREDICTIONS} \
--max-epochs 0 \
--inference-batch-size ${BATCH_SIZE} \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/baseline_${N_PREDICTIONS}preds_${BATCH_SIZE}batch" -append request_gpus=1 -append request_cpus=16
```

#### Experiments with 512 predictions

```bash
export FOLDER=2025-09-18-search-and-learn
export N_PREDICTIONS=64; condor_submit train.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions ${N_PREDICTIONS} \
--max-epochs 0 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${N_PREDICTIONS}i_baseline" -append request_gpus=1 -append request_cpus=32
```

```bash
export FOLDER=2025-09-18-search-and-learn
export INITIAL_PREDICTIONS=256
export EPOCHS=1
export PREDICTIONS_PER_EPOCH=256
export LEARNING_RATE=1e-5; condor_submit train_h100.condor command=" 
python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/search_and_learn_with_unsloth.py \
--initial-predictions ${INITIAL_PREDICTIONS} \
--predictions-per-epoch ${PREDICTIONS_PER_EPOCH} \
--learning-rate ${LEARNING_RATE} \
--max-epochs ${EPOCHS} \
--gpu_memory_utilization 0.5 \
--model-path /mnt/scratch/users/gbarbadillo/arc25/models/Llama-3.1-ARC-Potpourri-Induction-8B \
--dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/${FOLDER}/${INITIAL_PREDICTIONS}i_${EPOCHS}x${PREDICTIONS_PER_EPOCH}_lr${LEARNING_RATE}" -append request_gpus=1 -append request_cpus=32
```

### Local experiments

#### Debugging degradation of scores

I have observed a degradation in the number of valid outputs when using more than one batch. Let's run
some experiments to try to better understand the problem.

```bash
python scripts/search_and_learn_with_unsloth.py \
--initial-predictions 32 \
--inference-batch-size 8 \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-07-debug-unsloth-local/batch8

python scripts/search_and_learn_with_unsloth.py \
--initial-predictions 32 \
--inference-batch-size 8 \
--no-use-data-augmentation \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-07-debug-unsloth-local/no-data-augmentation-batch8-b
```

First experiments suggests that it is related to data augmentation.
I believe I have found and fixed the bug. It seemed that I was applying data augmentation
over and over on the same task, thus losing the traceability of the applied data augmentation.

#### Debug runtime

```bash
python scripts/search_and_learn_with_unsloth.py \
--initial-predictions 32 \
--inference-batch-size 8 \
--max-epochs 3 \
--predictions-per-epoch 32 \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/mini-arc-agi_evaluation_challenges.json \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-08-debug-runtime/batch8_32preds_3epochs

```

### How trl GRPOTrainer works

- https://huggingface.co/docs/trl/en/vllm_integration
- https://github.com/huggingface/trl/blob/659d2c1284e06862efbbccf64cd4310bcee4f200/trl/trainer/grpo_trainer.py#L54
- https://github.com/huggingface/trl/blob/main/trl/extras/vllm_client.py#L46
- https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_config.py#L22
- https://chatgpt.com/share/68c050f0-2ecc-8012-831d-29f7084ae526
- https://huggingface.co/learn/llm-course/en/chapter12/6

> When using vLLM, ensure the GPUs assigned for training and generation are separate to avoid NCCL communication conflicts.

- It seems that trl implements a custom VLLM that allows changing the weights.
- Examples using unsloth and GRPO do not enable VLLM, maybe GRPO patches it to use fast_generate.
- Trl code is very long, covering a lot of edge cases and difficult to understand

### OOM errors on Kaggle

When training on the longer tasks I'm getting OOM errors on Kaggle (24GB GPU).

```bash
[rank0]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 248.00 MiB. GPU 0 has a total capacity of 22.28 GiB of which 33.38 MiB is free. Process 6314 has 22.23 GiB memory in use. Of the allocated memory 20.60 GiB is allocated by PyTorch, with 199.88 MiB allocated in private pools (e.g., CUDA Graphs), and 171.43 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

### Making code execution robust

This was the problem. Sometimes execution hangs and no exception is thrown:

- https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/0iswo84s/logs
- https://wandb.ai/guillermobarbadillo/2025-09-11-search-and-learn/runs/xldcleic/logs

Sometimes raises exception:

- https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/kd4qttau/logs

Thus I have made many code changes to improve robustness, following suggestions by [ChatGPT](https://chatgpt.com/share/68c3ae7d-9cb4-8012-9950-9dd93606283e).

On my pc it executes very fast: `400/400 [00:02<00:00, 152.92pred/s]`

But in the cluster I'm seeing very slow executions:

- `12800/12800 [50:22<00:00,  4.24pred/s]`  [Experiment](https://wandb.ai/guillermobarbadillo/2025-09-12-search-and-learn/runs/19gni2he/logs)
- `51200/51200 [03:28<00:00, 245.29runs/s]` [Older experiment with good speed](https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/zdkkfzdv/logs)

However in Kaggle is also fast: `960/960 [00:03<00:00, 265.18pred/s]`

#### Reverting to old code

I have tried reverting back to commit 1557726a0e184d1a4e0b0490eec44bde7dde304e, from 8 september when I logged fast execution times. However the problem persisted:

- 4 cpus -> 41.67runs/s
- 8 cpus -> 61.31runs/s
- 20 cpus -> 56.75runs/s
- 64 cpus -> 9.51runs/s
- 128 cpus -> 9.41 runs/s

I have also tried running on other machine (calculon19 instead of calculon21) but did not get better results:

- 8 -> 74.22runs/s
- 16 -> 86.01runs/s

#### Simpler script

Iterations have been slow because I'm doing inference with the model first. That makes that each
execution takes around 30 minutes. I need to create a script that allows me to see results much faster.
That way I will run the same script with the same data in the different settings and get more information
about the problem faster.

I have prepared the script and I cannot understand the problem. Could it be a problem with the environment?
TODO: repeat experiments when updating the environment

```bash
export N_CPUS=8; condor_submit train.condor command=" python /mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/debug_parallel_execution.py \ --dataset-path /mnt/scratch/users/gbarbadillo/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json \ --prediction-path /mnt/scratch/users/gbarbadillo/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_02_05_36_40_predictions.json" -append request_cpus=${N_CPUS} -append request_gpus=0
```

#### The code isn't robust yet

I have done a quick test trying to evaluate around ~6000 predictions per task and I have seen that
the code hangs.

If the evaluations are done sequentially, file by file, there is no problem except for this two files that produce consistent hangs:

- /mnt/hdd0/Kaggle/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_08_31_09_47_48_predictions.json, 252
- /mnt/hdd0/Kaggle/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_09_01_13_46_42_predictions.json, 670

Each file has 3200 tasks, so I could use that as reference.

Investigating those tasks I see that both tasks have a general except clause that might be causing
the problems:

```python
while True:
    try:
        expand_star(center_x, center_y, star_color, center_color, distance)
        distance += 1
    except:
        break
```

After fixing that I have been able to evaluate the 6064 predictions per task in one run without problems.

## Results

### Unsloth/VLLM inference throughput

![alt text](res/1756991831223_image.png)

Making more predictions has higher throughput, also using a bigger batch size has higher throughput but at the cost of lowering prompt diversity.
I would need to tune this hyperparameters.

### First experiments

https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn

- Execution time with two partitions is 6x the baseline (5300s vs 31000)
- Each search and learn iteration is taking around 72 seconds. But half of the time seems to go prepare the model for training. I need to log that.
- Training takes around 20s per task
- Inference around 10s per taks, but is twice as slower as when doing the inference of all the tasks together
- Thus I estimate that an efficient implementation could do the same in 12000 
seconds (`30*400`).

```bash
#Baseline just inference
Processed prompts: 100%|██████████| 51200/51200 [1:17:57<00:00, 10.95it/s, est. speed input: 21508.87 toks/s, output: 4414.70 toks/s]


https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/xceyyl8q/logs
#training
{'train_runtime': 19.3244, 'train_samples_per_second': 2.432, 'train_steps_per_second': 2.432, 'train_loss': 0.2810402642539207, 'epoch': 1.0}
{'train_runtime': 23.5279, 'train_samples_per_second': 1.488, 'train_steps_per_second': 1.488, 'train_loss': 0.25343174253191264, 'epoch': 1.0}
{'train_runtime': 18.2541, 'train_samples_per_second': 1.972, 'train_steps_per_second': 1.972, 'train_loss': 0.27254368571771515, 'epoch': 1.0}

#inference
Processed prompts: 100%|██████████| 64/64 [00:10<00:00,  5.94it/s, est. speed input: 6103.78 toks/s, output: 2366.72 toks/s]
Processed prompts: 100%|██████████| 64/64 [00:11<00:00,  5.53it/s, est. speed input: 15778.11 toks/s, output: 2169.40 toks/s]
Processed prompts: 100%|██████████| 64/64 [00:12<00:00,  5.10it/s, est. speed input: 9888.11 toks/s, output: 2197.39 toks/s]

Tasks:  70%|██████▉   | 279/400 [5:23:04<2:25:52, 72.33s/task]2025-09-08 06:15:54,100 - __main__ - INFO - main -

#30 seconds seem to be training startup, half of the time.

# effect of the number of search and learn steps: [1, 3, 7]
Tasks:  70%|██████▉   | 279/400 [5:23:04<2:25:52, 72.33s/task]2025-09-08 06:15:54,100 - __main__ - INFO - main -

Tasks:   6%|▋         | 26/400 [57:03<13:05:33, 126.02s/task]2025-09-08 13:59:13,056 - __main__ - INFO - main - Search and learn for task 136b0064
Tasks:   3%|▎         | 12/400 [44:55<24:35:11, 228.12s/task]2025-09-08 13:56:54,181 - __main__ - INFO - main - Search and learn for task 0a2355a6


```

| learning rate            | is_correct | valid outputs | unique outputs |
|--------------------------|------------|---------------|----------------|
| baseline (no finetuning) | 16.90%     | 70.80%        | 49.50%         |
| 1.00E-03                 | 14.75%     | 35.50%        | 28.11%         |
| 1.00E-04                 | _17.50%_   | **74.91%**    | **51.01%**     |
| 1.00E-05                 | **18.25%** | _72.42%_      | 49.18%         |
| 1.00E-06                 | 17.25%     | 68.33%        | 47.89%         |
| 1.00E-07                 | 16.00%     | 70.98%        | _49.70%_       |

First experiments show a small but noticeable improvement when doing search and learn. Notice that
only one iteration of search and learn was done. So the baseline just did 128 predictions, and the
other experiments did 64 predictions, learned from those and did 64 additional predictions with the
finetuned model.

### Analyze inefficiencies in approach

The total amount of compute should be independent of the number of search and learn iterations that I
do per task, but the runtime is being heavily affected:

- 1 iteration: 72s/task
- 3 iterations: 126s/task
- 7 iterations: 228s/task

Let's analyze why this is happening

```bash
# 3 iterations 0b17323b, Total: 122s
Reset PEFT weights: 5s
Prepare training data: 3s
Training startup: 14s
Train: 15s
Inference: 7s
Prepare training data: 2s
Training startup: 15s
Inference: 7s
Prepare training data: 2s
Training startup: 16s
Train: 15s
Inference: 9s
Total: 122s
# 7 iterations 0b17323b, Total 245s
Reset PEFT weights: 6s
Prepare training data: 1, 2, 2
Training startup: 10, 10, 11
Train: 7, 7, 7
Inference: 8, 11, 7
# 1 iteration ff72ca3e, 86s
Reset PEFT weights: 10
Prepare training data: 5
Training startup: 22
Train: 27
Inference: 14
```

Training startup time is not constant, that is weird. Maybe I'm not measuring it correctly and it has
something to do with the data.

This is a summary table for the 3 iterations.

* **Total time (all entries): 125 s** (2 min 2 s)

| Task                  | Count | Average time |
| --------------------- | ----: | -----------: |
| Reset PEFT weights    |     1 |       5.00 s |
| Prepare training data |     3 |       2.33 s |
| Training startup      |     3 |      15.00 s |
| Train                 |     3 |      15.00 s |
| Inference             |     3 |       7.67 s |

I need to do more experiments and better log the execution time.

#### Tokenize before training

On a local experiment I have been able to reduce the execution time from 960s to 914s by tokenizing the dataset before training.

I have the feeling that training startup at H100 is longer than 3090, and also since training and inference on 3090 is around 4 times slower than on the H100, the startup time has a smaller effect.

It has taken around 228s per task on the 3090 (although I only used 4 tasks.) I was doing 128 predictions, so I could likely be doing 512 predictions on Kaggle.

### L4 is slower than I thought, half as fast as the 3090

When making inference for the ARC-AGI-2 evaluation set I get around 340 token/s of throughput with Kaggle's L4 GPUs. In comparison I can get around 650 token/s with my 3090 GPUs. So L4 is around half as fast as the 3090.

Notice that the 3090 was launched in September 2020, and the L4 was launched on March 2023. Quite surprising.

On average each prediction is taking around 1.26 seconds on Kaggle. That implies that **I won't be able to do more than 512 predictions per task on a submission using the current model.** And that is without considering
the training time, that would be the time for just making predictions. A more conservative approach
would be 256 predictions per task, or even less. Thus we need a much stronger model than the BARC one
to be able to reach 85% accuracy.

### Is the current implementation efficient enough?

#### Kaggle L4

TODO: I'm currently running tests on Kaggle to measure GPU usage and throughput. https://docs.google.com/spreadsheets/d/1NmmCZA7gPOyoBypwvpw_JhYdjcvqNFHibX_WahwTHIM/edit?gid=0#gid=0&range=A783

#### H100

TODO: 128 preds

TODO: 512 preds

### Do I have clear evidence that the approach works?

TODO: Kaggle
TODO: Cluster

## Conclusion

TODO:

- Do I have clear evidence that the approach works?
- Is the current implementation efficient enough?

## Next steps

- After seeing that throughput at inference increases with the number of predictions, I might have to
  use VLLM as a server and make async calls in parallel.

## TODO

- [x] Try unsloth for both training and inference
- [x] Compare unsloth speed against trl and VLLM
- [x] Create a smaller version of the dataset for faster experimentation
- [ ] Move code to script
  - [x] Move current notebook to script
  - [x] Refactor
  - [x] Move code to library modules
  - [x] Save results to disk
  - [x] Log to wandb.
    - [x] Tables, runtime...
    - [x] Task evolution
    - [x] Summary
    - [x] The goal is to be able to compare runs very easily with wandb. And also ideally to diagnose hyperparameter problems.
  - [x] All parameters should be on the configuration
  - [x] Log search vs learn time
  - [ ] Only save original outputs for test (that's what I need for the submission)
- [x] Try flashinfer and check if there is any speedup: https://github.com/flashinfer-ai/flashinfer
  - `pip install flashinfer-python`
  - FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'
  - Tried with prebuilt wheel but freezes when starting inference. `pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl#sha256=43d767b912c0c43a04be99595e0123eab9385fc72530a2874b5fb08e3145c0be
Collecting flashinfer-python==0.2.5+cu124torch2.6`
  - Should revisit on a future iteration because it could give faster inference for free
- [ ] Check the lora modules parameters, I'm using them without understanding
- [ ] Learning rate sweep. Using a small learning rate should be equivalent to just doing search. Using a too big lr should result in degraded metrics. There should be a sweet spot.
- [x] Code execution is not robust.
  - [x] Sometimes execution hangs and no exception is thrown
      - [x] https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/0iswo84s/logs
      - [x] https://wandb.ai/guillermobarbadillo/2025-09-11-search-and-learn/runs/xldcleic/logs
  - [x] Sometimes raises exception
    - [x] https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/kd4qttau/logs
  - [x] Made many changes to improve robustness: https://chatgpt.com/share/68c3ae7d-9cb4-8012-9950-9dd93606283e
- [ ] Investigate the time lost on training startup
- [ ] Experiment on Kaggle
  - [x] Upload the model. https://www.kaggle.com/models/ironbar/barc0llama-3.1-arc-potpourri-induction-8b
  - [x] Upload the code. https://www.kaggle.com/datasets/ironbar/arc25-source-code
  - [x] Create a notebook with the requirements. https://www.kaggle.com/code/ironbar/search-and-learn
  - [x] Split the data in 4, each for a GPU
  - [x] Collect the results to make a submission
  - [ ] How efficient is the current implementation?
  - [x] Getting OOM cuda errors when training on the longer tasks
  - [x] Create python module to do the submission, with tests
  - [x] Need a way to evaluate the submission once it's created
  - [x] Disable internet
  - [x] Implement dry run
  - [ ] Can I leave logs and keep making submissions?
  - [x] Speed seems to be lower than 3090, `1.26s/it, est. speed input: 2163.11 toks/s, output: 343.46 toks/s`
    - [x] Speed test on 3090.
      - [ ] 960 preds,  1.23it/s, est. speed input: 3389.69 toks/s, output: 547.71 toks/s
      - [ ] 960 preds unquantized, 1.42it/s, est. speed input: 3943.22 toks/s, output: 652.53 toks/s
      - [ ] 3840 preds, 1.47it/s, est. speed input: 4081.55 toks/s, output: 650.98 toks/s
      - [ ] Yes, seems to be around twice as fast
    - [x] 3x https://technical.city/en/video/GeForce-RTX-3090-vs-L4
    - [x] 2.3x https://chatgpt.com/share/68c1d348-89a4-8012-8135-a58a82bbef4d
    - [x] With current implementation I won't be able to make more than 512 predictions on a submission
- [ ] Check implementation of RL and how it alternates between training and inference(trl, GRPO)
- [ ] Analyze clusters results with 128 predictions. 
  - [ ] Learning rate
  - [ ] efficiency
  - [ ] improvements
- [ ] GPU efficiency and throughtput experiments on Kaggle
- [ ] Run experiments in the cluster with 512 predictions.
- [ ] How to improve efficiency? Maybe on a different iteration.
  - [ ] Do not train on all the cases
  - [ ] Remove duplicates
  - [ ] Filter cases with lower scores (as I did)
  - [ ] How fast is inference compared to training?
  - [ ] Train for multiple epochs
  - [ ] LoRA parameters
