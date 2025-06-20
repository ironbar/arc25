# Iteration 12. Solve a few ARC tasks

_17-06-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Probe that I can solve a few selected ARC tasks by using an LLM to write code.

## Motivation

On the previous [Iteration 10](Iteration_10_solve_arc_tasks.md) I tried to solve a few ARC tasks without
success: `08ed6ac7, 0b148d64, 0ca9ddb6, 0d3d703e, 178fcbfb, 1bfc4729, 1c786137`. The goal of this iteration
is to solve all those tasks by implementing new training tasks and/or improving the solving algorithm.

I should avoid creating training tasks that are clones from the real ARC tasks, otherwise I cannot
measure the generalization capability of the model. My goal should be to write training tasks that
teach the core knowledge that is needed for ARC.

## Development

### New tasks to implement

- [ ] Sort objects and do something to them based on the order. I can sort objects based on: area, x, y. I can move the objects, change their colors. This requires more control over the input images.
- [ ] Learn to use the color of the object. Let's focus on monochrome objects by now. Based on the color of the object something is done (move, change color, crop)
- [ ] Aggregate properties and use them to select, f.e. most/least popular color/area/shape...
- [x] Learn to draw using object center as a reference, points, lines (also vertical and horizontal), rectangles...
- [x] Create more tasks with apply_colormap
- [x] Learn to draw using color of the objects as a reference
- [ ] More tasks about selecting an object that has some unique or extreme property

### Training

<details>
  <summary>Click to expand/collapse this section</summary>

```bash
export N_GPUS=2
export PARAMETERS=0.5B
export LEARNING_RATE=1e-4
export STEPS=2000; condor_submit train.condor command="
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning.py \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-18-more-training-tasks/${N_GPUS}xA6000-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${LEARNING_RATE}lr \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 2 \
--per-device-eval-batch-size 4 \
--batch-size 32 \
--max-seq-len 6144 \
--logging-steps 10 \
--eval-steps 50 \
--save-steps 200 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

export N_GPUS=2
export PARAMETERS=1.5B
export LEARNING_RATE=1e-4
export STEPS=2000; condor_submit train.condor command="
accelerate launch --num_processes ${N_GPUS} --num_machines 1 --mixed_precision bf16 --multi_gpu  \
/mnt/scratch/users/gbarbadillo/arc25/arc25/scripts/finetuning.py \
--model_path /mnt/scratch/users/gbarbadillo/arc25/models/Qwen2.5-Coder-${PARAMETERS}-Instruct/ \
--output-dir /mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-18-more-training-tasks/${N_GPUS}xA6000-Qwen2.5-Coder-${PARAMETERS}-${STEPS}steps-${LEARNING_RATE}lr \
--device-map None \
--max-steps ${STEPS} \
--n-gpus ${N_GPUS} \
--learning-rate ${LEARNING_RATE} \
--per-device-train-batch-size 1 \
--per-device-eval-batch-size 2 \
--batch-size 32 \
--max-seq-len 6144 \
--logging-steps 10 \
--eval-steps 50 \
--save-steps 200 \
--lora-r 32 \
--use-dora \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

rsync -P -r calculon01:/mnt/scratch/users/gbarbadillo/arc25/trainings/2025-06-18-more-training-tasks /mnt/data/MEGA/TEMP --exclude wandb/* --exclude *.pt
```

</details>

## Results

### Influence of training steps and diversity of predictions

When making predictions with a model trained for 8k steps I was surprised to see that only produced 1 unique
prediction (making a ton of repeated predictions)

| task \ steps (k) | 1   | 2   | 4   | 8  | 16 |
|------------------|-----|-----|-----|----|----|
| 1bfc4729         | 2   | 16  | 10  | 1  | 14 |
| 0ca9ddb6         | 69  | 31  | 20  | 17 | 11 |
| 178fcbfb         | 5   | 10  | 5   | 8  | 6  |
| 0d3d703e         | 121 | 120 | 115 | 14 | 88 |

The table below shows the number of unique and valid predictions for the `Qwen2.5-Coder-0.5` model.
The total number of predictions was 136.
The relation is unclear and inconsistent between tasks.

Thus so far does not seem that training for longer reduces the model predictions diversity.

## Conclusion

## Next steps

## TODO

- [ ] Write new training tasks to solve the current knowledge gaps of the model
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.
- [ ] If possible I should use Kaggle compute for evaluation. It is almost free and is a good way to store and visualize results.
- [ ] Compositionality, can the model solve the task that selects the biggest object, crop and trim? That
  would be a good example of compositionality because those functions were not used together in the dataset
- [ ] Sequential solving. Try also solving the tasks in multiple steps, not just once. It could help
  with compositionality.
