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
export STEPS=16000; condor_submit train.condor command="
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

export N_GPUS=2
export PARAMETERS=3B
export LEARNING_RATE=1e-4
export STEPS=16000; condor_submit train.condor command="
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
--use-liger-kernel \
--use-rslora" -append request_gpus=${N_GPUS} -append request_cpus=8

export N_GPUS=2
export PARAMETERS=7B
export LEARNING_RATE=1e-4
export STEPS=16000; condor_submit train.condor command="
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
--max-seq-len 5120 \
--logging-steps 10 \
--eval-steps 50 \
--save-steps 200 \
--lora-r 32 \
--use-dora \
--use-liger-kernel \
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

### Analysis of trying to solve the tasks

| task \ model | 0.5B@1k steps                                                                            | 1.5B@1k steps                                                                                                    | 1.5B@16k steps                                                                             | 3B@1k steps                                                                                          | 7B@1k steps                                                                             |
|--------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| 08ed6ac7     |                                                                                          |                                                                                                                  |                                                                                            | does not understand that the task is about changing colors, sorting the objects by area              | does not understand that the task is about changing colors, sorting the objects by area |
| 0b148d64     |                                                                                          |                                                                                                                  |                                                                                            | the most succesfull approach is downscaling instead of selecting and cropping                        | OOM                                                                                     |
| 0ca9ddb6     | draws 3 points, tries to use area for color. Tried an attempt to use the color as input  | draws 2 points, tries to use area for color                                                                      | draws 2 points, tries to use area for color                                                | draws 4 points, but doesn't understand that color depends on the object color, tries to use the area | draws 3 points, then starts to draw lines                                               |
| 0d3d703e     | does not understand that  is about colormaps                                             | does not understand that is about colormaps                                                                      | Solved at epoch 6                                                                          | Solved at epoch 2                                                                                    | Solved at epoch 3                                                                       |
| 178fcbfb     | draws vertical or horizontal lines, but not both                                         | draws vertical and horizontal lines, but does not understand there is a condition                                | only vertical lines, very low diversity                                                    | draws vertical and horizontal lines, but does not understand there is a condition                    | draws vertical and horizontal lines, but does not understand there is a condition       |
| 1bfc4729     | only horizontal lines                                                                    | only horizontal lines                                                                                            | does not understand the task, draws horizontal lines on the points and the rest is garbage | low diversity in predictions, does not improve over horizontal lines                                 | many different predictions, but not in the correct direction                            |
| 1c786137     |                                                                                          | chooses the object using height instead of area, maybe another property is needed. Probably color should be used |                                                                                            | does not understand the task                                                                         | OOM                                                                                     |

### Thoughts

- I have the feeling that bigger models do better
- I have solved the first real ARC task, although it was very simple it required adaptation with HER
- But the lack of generalization is worrying, maybe the training data generation strategy is not the best
- Lack of creativity, only does what it has learned to do during training
- HER works, but needs a model with diverse predictions and good intuition

If the model is in the right direction, I believe it's very likely that HER will help to achieve the
correct solution. However so far the model is lacking that ability to understand the tasks and use
the appropriate DSL primitives to solve the problem.

Another problem is the low diversity in the proposed solutions. For some tasks-model combinations it is
as low as proposing the same solution over and over. Reinforcement learning requires exploration to
solve a problem, and in many cases the solution space is not being explored correctly.

Deep learning works when the training set densely covers the space. That is not the case for the current
training tasks. It was the case for the toy drawing problem, because the space was small. However when the
DSL grows that becomes more and more difficult.

## Conclusion

On this iteration I have prepared new sample tasks to learn how to use the DSL. Despite of doing this
job only one real ARC task was solved (and it was simply applying a colormap). 

I have to rethink the approach, because the current implementation does not correctly explore the solution space. 
Only explores a small fraction of the solution space and repeats the same errors over and over.

## Next steps

- Better sampling strategy. Could play with temperature, top_k and top_p to create more diverse samples. https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/text_generation#transformers.GenerationConfig.temperature
- Better training objective. label_smoothing_factor might be used to preserve entropy. https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/trainer#transformers.TrainingArguments.label_smoothing_factor
- Validation might be solved ARC tasks. That way I could better measure the effect of the training tasks.
- Reread transduction and induction paper, and code.
- What if I give hints of how to solve the problem? Is the model capable on that case?

## TODO

- [x] Write new training tasks to solve the current knowledge gaps of the model
- [ ] I need a way to do evaluation at scale, using multiple GPUs, and saving all the generated tasks when searching for a solution.
- [ ] If possible I should use Kaggle compute for evaluation. It is almost free and is a good way to store and visualize results.
- [ ] Compositionality, can the model solve the task that selects the biggest object, crop and trim? That
  would be a good example of compositionality because those functions were not used together in the dataset
- [ ] Sequential solving. Try also solving the tasks in multiple steps, not just once. It could help
  with compositionality.
