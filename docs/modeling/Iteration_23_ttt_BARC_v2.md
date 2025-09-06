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

## Results

### Unsloth/VLLM inference throughput

![alt text](res/1756991831223_image.png)

Making more predictions has higher throughput, also using a bigger batch size has higher throughput but at the cost of lowering prompt diversity.
I would need to tune this hyperparameters.

## Conclusion

## Next steps

- After seeing that throughput at inference increases with the number of predictions, I might have to
  use VLLM as a server and make async calls in parallel.

## TODO

- [x] Try unsloth for both training and inference
- [x] Compare unsloth speed against trl and VLLM
- [ ] Move code to script
  - [x] Move current notebook to script
  - [ ] Refactor
  - [x] Move code to library modules
  - [ ] Save results to disk
  - [ ] Log to wandb. Tables, runtime...
  - [ ] All parameters should be on the configuration
- [ ] Try flashinfer and check if there is any speedup: https://github.com/flashinfer-ai/flashinfer
  - `pip install flashinfer-python`
  - FileNotFoundError: [Errno 2] No such file or directory: 'nvcc'
  - Tried with prebuilt wheel but freezes when starting inference. `pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl#sha256=43d767b912c0c43a04be99595e0123eab9385fc72530a2874b5fb08e3145c0be
Collecting flashinfer-python==0.2.5+cu124torch2.6`
  - Should revisit on a future iteration because it could give faster inference for free
- [ ] Check the lora modules parameters, I'm using them without understanding
