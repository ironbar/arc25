# Iteration 11. Pretrain LoRA on new tasks

_29-05-2025_

## Goal

Check if pretraining the LoRA on the new ARC25 tasks can bring improvements on the leaderboard.

## Motivation

While I'm working on the new approach for ARC25 I have had an idea to improve the current leaderboard score of 11.94: instead of starting from a random LoRA pretrain it on the new ARC25 tasks.

On my experiments during ARC24 I already noticed that it was better to fine-tune a pretrained LoRA instead of fine-tuning a fresh new LoRA. If that finding was correct, this experiment might work. Additionally there might be some shared information between the new tasks and the test sets.

## Development

### Create subsets from ARC25 with the new tasks

https://www.kaggle.com/code/ironbar/new-arc-agi-2-tasks

### Fine-tune LoRAs with the new tasks

### Check if results improve on the evaluation set

## Results

## Conclusion

## Next steps

## TODO

- [ ]
