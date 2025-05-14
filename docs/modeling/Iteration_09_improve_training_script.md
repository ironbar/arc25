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
- 

## Results

## Conclusion

## Next steps

- Solve the training set, then the evaluation set, then the new tasks from ARC25.

## TODO

- [x] Fix the problem with repeated calls to the train dataset generator
- [ ] Make the script work with accelerate
- [ ] Measure training speed vs input size
- [ ] Enable multi-task training, currently only trains on a single task
- [ ] Measure data sampling speed to verify is fast enough
- [ ] Add validation
