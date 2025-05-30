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

I have to check the original fine-tuning parameters used by the architects. Probably use a batch size
bigger than one because I'm training on multiple tasks.
The idea is to create a new notebook to do this task. Probably the most efficient implementation is to
train a different LoRA rank on each GPU, thus each experiment will produce 4 LoRAs.

- https://www.kaggle.com/code/ironbar/pretrain-loras-for-the-architects
- https://www.kaggle.com/code/ironbar/the-architects-baseline-with-4-gpus

These are the parameters that I used on my script for 4 GPUs, I changed the default parameters from
the architects, but they can be seen as comments.

```python
max_seq_length_train = 8192 # default 4224

per_device_train_batch_size=4, # default=2
gradient_accumulation_steps=1, # default=2
learning_rate=1e-4,
embedding_learning_rate=1e-5,
```

### Check if results improve on the evaluation set

I have to prepare a new notebook that uses pretrained LoRA. As far as I see, I simply have to provide the path to the pretrained LoRA when loading the model. I could remove all the peft configuration completely.

## Results

## Conclusion

## Next steps

## TODO

- [x] Check where random seed is used. Seed is on `infer_aug_params`
