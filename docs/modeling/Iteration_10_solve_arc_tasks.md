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

## Results

## Conclusion

## Next steps

- Hypothesis: If I implement a DSL that covers the whole training and evaluation set, it should generalize to the test set.

## TODO

- [x] Add safety and determinism checks
- [x] Add more primitive functions and training tasks to learn to use them
- [x] I would like to have a list of all the primitive functions from the DSL, and how many times are they used in the training tasks. A correlation plot would also be nice to see which connections are missing.
- [ ] Is the sampling speed enough?
- [ ] Stats about the input tokens distribution, what should be the max-seq-len?
- [ ] Optimize learning rate and batch size for 2 GPUs.
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.