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

#### Cluster

```bash
export N_GPUS=2
export PARAMETERS=0.5B
export STEPS=16000
condor_submit train.condor command="
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning.py \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-10-first-real-trainings/A6000-GPUS${N_GPUS}-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--per-device-train-batch-size 4 \
--per-device-eval-batch-size 8 \
--batch-size 32 \
--max-seq-len 3072 \
--logging-steps 100 \
--eval-steps 0 \
--save-steps 1000 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=12
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

# adding this other line
import multiprocessing as mp, os
print(">>> multiprocessing start-method:", mp.get_start_method(), "PID:", os.getpid())
# local response
>>> multiprocessing start-method: fork PID: 19840
>>> multiprocessing start-method: fork PID: 19841
```

#### Local experiments

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

To be safe I should probably use `max-seq-len=8192`.

## Results

## Conclusion

## Next steps

- Hypothesis: If I implement a DSL that covers the whole training and evaluation set, it should generalize to the test set.

## TODO

- [x] Add safety and determinism checks
- [x] Add more primitive functions and training tasks to learn to use them
- [x] I would like to have a list of all the primitive functions from the DSL, and how many times are they used in the training tasks. A correlation plot would also be nice to see which connections are missing.
- [x] Is the sampling speed enough?
- [x] Stats about the input tokens distribution, what should be the max-seq-len?
- [ ] Optimize learning rate and batch size for 2 GPUs.
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.
