# Iteration 9. Improve training script

_14-05-2025_

## Goal

Improve the training script so I can start working towards solving real ARC tasks with code.

## Motivation

I have seen that Hindsight Experience Replay (HER) allows to generalize to novel tasks. Next step is
to probe that it can solve real ARC tasks, not just toy tasks. But previously I have to make some updates
to the training script. That will allow me to iterate faster on the next steps.

## Development

### Fix the problem with repeated calls to training generator

```bash
conda activate arc25
export CUDA_VISIBLE_DEVICES=0
python finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_training_generator --device-map auto --random-seed 5 --max-steps 11 --n-gpus 1 --per-device-train-batch-size 1 --batch-size 16 --max-seq-len 1024 --no-log-to-wandb --no-resume-from-checkpoint
```

- [IterableDataset](https://huggingface.co/docs/datasets/v3.6.0/en/package_reference/main_classes#datasets.IterableDataset)
- [SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTTrainer)

It seems that it is the expected behaviour, however I have modified the generator to just yield samples.
The setting of the random seed and printing the first sample is now outside.

### Make the script work with accelerate

```bash
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 --multi_gpu \
finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 40 --n-gpus 2 --per-device-train-batch-size 8 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint
```

I'm using the latest version of accelerate: `1.6.0`, the thing is that previously the `SFTConfig` class
had a `dispatch_batches=False` parameter that now is missing.

- https://huggingface.co/docs/accelerate/en/package_reference/accelerator
- https://huggingface.co/docs/accelerate/v1.6.0/en/package_reference/utilities#accelerate.DataLoaderConfiguration
- https://github.com/huggingface/transformers/issues/34699
- https://huggingface.co/docs/transformers/v4.51.3/en/main_classes/trainer#transformers.TrainingArguments

The solution was easy, but difficult to find: `accelerator_config=dict(dispatch_batches=False`

### Training speed test

By using 2 GPUs and the right batch size we can improve the training speed by a factor of 5.

| Number of GPUs | Per Device Batch Size | Train Samples per Second |
|----------------|------------------------|---------------------------|
| 2              | 8                      | 44.25                     |
| 1              | 8                      | 25.69                     |
| 1              | 4                      | 22.27                     |
| 1              | 2                      | 14.70                     |
| 1              | 1                      | 8.85                      |

```bash
accelerate launch --num_processes 2 --num_machines 1 --mixed_precision bf16 --multi_gpu \
finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 2 --per-device-train-batch-size 8 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
# {'train_runtime': 36.1583, 'train_samples_per_second': 44.25, 'train_steps_per_second': 2.766, 'train_loss': 0.2923687481880188, 'epoch': 1.0}
export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 \
finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 1 --per-device-train-batch-size 8 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
# {'train_runtime': 63.3117, 'train_samples_per_second': 25.272, 'train_steps_per_second': 1.579, 'train_loss': 0.2931043267250061, 'epoch': 1.0}
python finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 1 --per-device-train-batch-size 8 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
# {'train_runtime': 62.2894, 'train_samples_per_second': 25.687, 'train_steps_per_second': 1.605, 'train_loss': 0.29407034754753114, 'epoch': 1.0}
python finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 1 --per-device-train-batch-size 4 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
#{'train_runtime': 71.8484, 'train_samples_per_second': 22.269, 'train_steps_per_second': 1.392, 'train_loss': 0.29404119253158567, 'epoch': 1.0}
python finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 1 --per-device-train-batch-size 2 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
#{'train_runtime': 108.8354, 'train_samples_per_second': 14.701, 'train_steps_per_second': 0.919, 'train_loss': 0.29236586928367614, 'epoch': 1.0}
python finetuning.py --output-dir /mnt/hdd0/Kaggle/arc25/trainings/20250514/debug_accelerate --device-map None --random-seed 5 --max-steps 100 --n-gpus 1 --per-device-train-batch-size 1 --batch-size 16 --max-seq-len 512 --no-log-to-wandb --no-resume-from-checkpoint --save-steps 100
# {'train_runtime': 180.7981, 'train_samples_per_second': 8.85, 'train_steps_per_second': 0.553, 'train_loss': 0.29323326468467714, 'epoch': 1.0}
```

## Results

## Conclusion

## Next steps

- Solve the training set, then the evaluation set, then the new tasks from ARC25.

## TODO

- [x] Fix the problem with repeated calls to the train dataset generator
- [x] Make the script work with accelerate
- [x] Measure training speed vs batch size and number of gpus
- [ ] Measure training speed vs input size
- [ ] Enable multi-task training, currently only trains on a single task
- [ ] Measure data sampling speed to verify is fast enough
- [ ] Add validation
