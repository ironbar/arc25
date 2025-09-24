# Iteration 27. Improve search and learn

_start date_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

We already know that search and learn works. Can we make it faster and better?

## Motivation

## Development

### Speed analysis of all the parts

TODO: the goal is to measure where the execution time is going so I can have more information to design improvements

- Probably the easiest way to do it is to use the logs from Kaggle.
- I can also use the logs from the cluster experiments

### Local experiments with group sizes

```bash
export GROUP_SIZE=1
python scripts/search_and_learn_with_unsloth.py \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small_arc-agi_training_challenges.json \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-24-debug-grouped-tasks/group-size-${GROUP_SIZE} \
--task-group-size ${GROUP_SIZE} \
--max-epochs 3 \
--initial-predictions 8 \
--predictions-per-epoch 8

export GROUP_SIZE=67
python scripts/search_and_learn_with_unsloth.py \
--dataset-path /mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/small_arc-agi_training_challenges.json \
--output-dir /mnt/hdd0/Kaggle/arc25/trainings/2025-09-24-debug-grouped-tasks/group-size-${GROUP_SIZE} \
--task-group-size ${GROUP_SIZE} \
--max-epochs 3 \
--initial-predictions 8 \
--predictions-per-epoch 8
```

## Results

## Conclusion

## Next steps

## TODO

- [ ] Maybe I should discard the idea of tuning for each task independently if the data shows that it is too inefficient
  - [ ] Edit the script to support grouping tasks
  - [ ] Compare speed of grouping vs no grouping
