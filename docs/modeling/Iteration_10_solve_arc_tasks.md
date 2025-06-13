# Iteration 10. Solve ARC tasks

_22-05-2025_

## Goal

Can I solve real ARC tasks with code and HER?

## Motivation

I have already seen that HER allows to generalize to novel toy tasks, I need to check if it can solve
real ARC tasks.

I know that my primitive functions defined on ARC24 solved 285 training tasks. So probably the easiest
path is to review those transformations add them and modify if needed.

## Development

### Add safety and determinism checks

Inspired by [Absolute Zero Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) I'm going
add safety and determinism checks.

### Generation functions

LLM are incredible useful to write generation functions. For example I have asked o3
to write a function to create ARC images with random objects and it worked perfectly.

### Stats about current implementation

```
Found 23 training tasks

There are 17 DSL functions defined in arc25.dsl:
	DSL functions used in 1000 tasks:
detect_objects                   621 times
draw_object                      425 times
create_img                       395 times
crop                             123 times
draw_rectangle                   122 times
draw_vertical_line               111 times
draw_horizontal_line              98 times
draw_line                         94 times
draw_pixel                        91 times
mode                              49 times
apply_colormap                    46 times
downscale                         33 times
rotate_90                         32 times
flip                              28 times
upscale                           27 times
pad                               17 times
trim                              15 times

There are 13 DSL attributes defined in arc25.dsl:
	DSL attributes used in 1000 tasks:
change_color                     380 times (Object)
area                              89 times (Object)
height                            72 times (BoundingBox, Object)
width                             66 times (BoundingBox, Object)
is_horizontal_line                48 times (Object)
is_rectangle                      46 times (Object)
move                              45 times (Object)
is_square                         42 times (Object)
is_vertical_line                  40 times (Object)
center                            38 times (Object)
is_line                           32 times (Object)
is_point                          31 times (Object)
copy                               0 times (Object)
```

This is clearly not enough, but I want to train a model on these tasks and see if it can solve any
of the ARC training tasks.

### Training

#### Local experiments

<details>
  <summary>Click to see the bash commands</summary>


```bash
export N_GPUS=2
export PARAMETERS=0.5B
export STEPS=10
export MAXSEQLEN=3072
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
scripts/finetuning.py \
--model_path /home/gbarbadillo/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-06-10-first-real-trainings/3090-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 2 \
--per-device-eval-batch-size 4 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora \
--no-resume_from_checkpoint


export MAXSEQLEN=8192
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
scripts/finetuning.py \
--model_path /home/gbarbadillo/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-06-10-first-real-trainings/3090-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 2 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora

export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export MAXSEQLEN=8192
python scripts/finetuning.py \
--model_path /home/gbarbadillo/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-06-10-first-real-trainings/3090-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 2 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora
```

</details>


It is training around 2.66s/it, task generation does not seem to be the bottleneck. At the beginning of the training used multiple cores to sample, but then CPU usage was low. Probably
the queue was filled.

```
# --max-seq-len 3072
{'train_runtime': 298.5646, 'train_samples_per_second': 10.718, 'train_steps_per_second': 0.335, 'train_loss': 0.24244340896606445, 'epoch': 1.0}
# export MAXSEQLEN=6144
{'train_runtime': 363.6891, 'train_samples_per_second': 8.799, 'train_steps_per_second': 0.275, 'train_loss': 0.23476869583129883, 'epoch': 1.0}
# export MAXSEQLEN=8192
2025-06-11 06:32:08,621 - __main__ - INFO - log_prompt_length_percentiles -     train number of prompts: 1000, max number of tokens : 5108, percentiles: {50: 1249, 75: 1567, 90: 1960, 95: 2069, 97: 2139}
{'train_runtime': 374.4698, 'train_samples_per_second': 8.545, 'train_steps_per_second': 0.267, 'train_loss': 0.23707759857177735, 'epoch': 1.0}

export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0
export MAXSEQLEN=8192
{'train_runtime': 671.3956, 'train_samples_per_second': 4.766, 'train_steps_per_second': 0.149, 'train_loss': 0.23653331756591797, 'epoch': 1.0} 
```

To be safe I should probably use `max-seq-len=8192`, otherwise we will be missing some training tasks.

#### Cluster

```bash
export N_GPUS=2
export PARAMETERS=0.5B
export LEARNING_RATE=2e-4
export STEPS=4000; condor_submit train.condor command="
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning.py \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-13-first-real-trainings/${N_GPUS}xA6000-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${LEARNING_RATE}lr \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 2 \
--per-device-eval-batch-size 4 \
--batch-size 32 \
--max-seq-len 8192 \
--logging-steps 10 \
--eval-steps 50 \
--save-steps 200 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=12

export N_GPUS=1
export PARAMETERS=0.5B
export STEPS=1000
export LEARNING_RATE=4e-5; condor_submit train.condor command="
python  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning.py \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-13-first-real-trainings/${N_GPUS}xA6000-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${LEARNING_RATE}lr \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 2 \
--per-device-eval-batch-size 4 \
--batch-size 32 \
--max-seq-len 8192 \
--logging-steps 10 \
--eval-steps 50 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS}
```


#### Debugging

<details>
  <summary>Click to expand/collapse this section</summary>

```bash

export N_GPUS=2
export PARAMETERS=0.5B
export STEPS=10
export MAXSEQLEN=8192
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
scripts/finetuning.py \
--model_path /home/gbarbadillo/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-06-10-first-real-trainings/3090-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 2 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora \
--no-resume_from_checkpoint

export N_GPUS=1
export PARAMETERS=0.5B
export STEPS=10
export MAXSEQLEN=8192
python \
scripts/finetuning.py \
--model_path /home/gbarbadillo/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-06-10-first-real-trainings/3090-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${MAXSEQLEN}msl \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 2 \
--batch-size 32 \
--max-seq-len ${MAXSEQLEN} \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora \
--no-resume_from_checkpoint \
--dataloader-num-workers 0

# this works --dataloader-num-workers 0
# this does not: --dataloader-num-workers 1
# it is unrelated from: os.environ['TOKENIZERS_PARALLELISM'] = 'true'
```

I'm seeing a new error on the cluster.

```
AttributeError: Can't pickle local object 'SFTTrainer._prepare_dataset.<locals>.add_eos'


# local libraries
accelerate                1.6.0                    pypi_0    pypi
torch                     2.6.0                    pypi_0    pypi
transformers              4.51.3                   pypi_0    pypi
datasets                  3.5.1                    pypi_0    pypi
trl                       0.18.0.dev0

# cluster libraries (experiments run on docker)
accelerate==1.7.0
torch==2.6.0
transformers==4.52.4
datasets==3.6.0
trl==0.18.1

# local experiments updating library versions
trl==0.18.0 -> works
trl==0.18.1 -> works
accelerate==1.7.0 -> works
transformers==4.52.4 -> works
datasets==3.6.0 -> works

# adding this line at the start of the script reproduces the problem locally
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
> [rank0]: AttributeError: Can't pickle local object 'SFTTrainer._prepare_dataset.<locals>.add_eos'

# adding this other line to see what it is printed
import multiprocessing as mp, os
print(">>> multiprocessing start-method:", mp.get_start_method(), "PID:", os.getpid())
# local response
>>> multiprocessing start-method: fork PID: 19840
>>> multiprocessing start-method: fork PID: 19841
# cluster response
>>> multiprocessing start-method: fork PID: 57
>>> multiprocessing start-method: fork PID: 58

# adding this line at the start does not solve the problem in the cluster
mp.set_start_method("fork", force=True)
```

- https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/data/dataloader.py#L173
- https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
- https://wandb.ai/guillermobarbadillo/2025-05-21-model-size/runs/3g8opphj/files/requirements.txt, on the last successful training in the cluster I used trl=0.17.0

Trying with trl=0.17.0 on job 204192. No success.

What is the problem? The problem seems to be that pickle cannot work with functions defined inside functions.
```
AttributeError: Can't pickle local object 'SFTTrainer._prepare_dataset.<locals>.add_eos'
AttributeError: Can't pickle local object 'truncate_dataset.<locals>.truncate'
```
So I have changed how the dataset is generator to avoid entering in that 'SFTTrainer._prepare_dataset.<locals>.add_eos' place, but then another similar error has arised.
The biggest question is why hasn't happened this before.

This is the most similar problem found online, but there is no solution: https://github.com/huggingface/trl/issues/2979

This guy says that spawn might be used by default sometimes: https://discuss.pytorch.org/t/the-default-value-of-dataloader-multiprocessing-context-is-spawn-in-a-spawned-process/107494

https://github.com/pytorch/pytorch/issues/44687 Slightly related.
Setting the DataLoader(..., multiprocessing_context='fork') fixes the issue for me.

This seems to solve the problem, change this line:

```
/home/gbarbadillo/miniconda3/envs/arc25-clone/lib/python3.10/site-packages/torch/utils/data/dataloader.py
multiprocessing_context=None,
multiprocessing_context='fork',
sed -i.bak "0,/multiprocessing_context[[:space:]]*=[[:space:]]*None,/s//multiprocessing_context='fork',/" \
/home/gbarbadillo/miniconda3/envs/arc25-clone/lib/python3.10/site-packages/torch/utils/data/dataloader.py

sed -i.bak "0,/multiprocessing_context[[:space:]]*=[[:space:]]*None,/s//multiprocessing_context='fork',/" \
/mnt/scratch/users/gbarbadillo/arc25/cached-environments/venv_07bdecf0b823319f4d2fcbe9cdc354d9/lib/python3.10/site-packages/torch/utils/data/dataloader.py
```
</details>

## Results

### Training Hyperparameters

For a batch size of 32 and lora rank of 32 a learning rate of 2e-4 seems to be good. 1e-3 is too much, 
4e-4 also works but for longer trainings 2e-4 might be a better option.

Since I only have 23 training tasks, I don't expect to see relevant improvements by using a batch size
bigger than 32. So I'm not going to do experiments with the batch size.

[Wandb experiment](https://wandb.ai/guillermobarbadillo/2025-06-13-first-real-trainings/workspace?nw=nwuserguillermobarbadillo), filter by `1000steps`.

### Fine-tuning capacity

TODO: change the lora rank and also try a full fine-tuning and check the training metrics

## Conclusion

## Next steps

- Hypothesis: If I implement a DSL that covers the whole training and evaluation set, it should generalize to the test set.

## TODO

- [x] Add safety and determinism checks
- [x] Add more primitive functions and training tasks to learn to use them
- [x] I would like to have a list of all the primitive functions from the DSL, and how many times are they used in the training tasks. A correlation plot would also be nice to see which connections are missing.
- [x] Is the sampling speed enough?
- [x] Stats about the input tokens distribution, what should be the max-seq-len?
- [x] Optimize learning rate and batch size for 2 GPUs.
- [ ] Create a notebook to evaluate the trained models on real ARC tasks
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.
