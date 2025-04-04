# Iteration 1. Architects baseline

_28-03-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

How far can we go using the Architects solution from ARC24?

## Motivation

ARC25 has just started and I believe establishing a baseline would be a good starting point.
Moreover I would like to know better the Architects' solution.

## Development

I have created my own [notebook](https://www.kaggle.com/code/ironbar/the-architects-baseline-with-4-gpus) that modifies the original solution to work with 4 GPUs.

### Hyperparameter tuning

#### min_prob

I have seen that increasing the value of min_prob results on faster inference, but at the same time I believe that less correct predictions are made. If I use `min_prob=None` all the predictions are correct, probably that uses a greedy approach whereas in the other case it only creates a prediction if the probability is higher than the parameter value.

```
# parameter values and inference times
min_prob=0.10 [63, 242, 277, 492]
min_prob=0.17 [35, 147, 195, 309]
min_prob=0.25 [23, 108, 154, 259]
min_prob=0.35 [17, 82, 118, 235]
min_prob=0.5  [13, 59, 66, 191]
```

#### Effect of `n`

The inference time is proportional to how many predictions we do per task.

```
n=1 [19, 77, 97, 162]
n=2 [35, 147, 195, 309]
```

#### Model mode

It does not have any effect. My guess is that I need a more recent version of unsloth. I should also add flash attention to be able to work with transformers models.

```
unsloth_4bit [35, 147, 195, 309]
unsloth_8bit [33, 151, 194, 335]
```

#### max_seq_length_train

By default it is set to `4224` but we could increase it to `8192` without having memory issues (60% VRAM). This will make training slower for the longest tasks.

#### Gradient checkpointing

I didn't see any significative change in training speed after disabling it.

### Better sorting algorithm

I have tried creating a better algorithm for sorting tasks between GPUs but at the end runtime was worse than the original algorithm. I have seen that training time is proportional to the training tokens, but inference time is not easy to predict. This might be explained by not knowing the output tokens and the depth first search algorithm used at inference.

### 2 inference runs per GPU

![](res/2025-04-04-13-56-14.png)

Looking at the GPU usage I noticed that when training GPU usage was 100%, but on inference many times it was below 50% (And memory usage was always below 50%). This opens the door for running two inference process on the same GPU.

On experiments I have seen a reduction of 30% of inference time (670s to 470s).

After the change GPU usage is almost 100%.

![](res/2025-04-04-13-59-23.png)

## Results

[Gsheet with results](https://docs.google.com/spreadsheets/d/1NmmCZA7gPOyoBypwvpw_JhYdjcvqNFHibX_WahwTHIM/edit?gid=0#gid=0&range=A1)

According to the official [ARC](https://arcprize.org/leaderboard) documentation the solution from the Architects should score around 2.5%, but my first successful submission scored 7%. I guess the improvement comes from splitting the data in 4 folds and using 4 gpus (instead of 2 folds on 2 gpus).

## Conclusion

## Next steps

- Probably the easiest way to keep improving the baseline is to split the data in more folds. Using 8 folds would naturally fit with using 8 processes at inference. I would have to see how to coordinate the training runs.
- Everyday we are allowed to make submissions with a compute cost of 50$. Over 7 months that will be around 10k$. That is a lot of money, maybe I should use the private test set as my development set and use the evaluation set as the test set (to decide the final submission).

## TODO

- [ ] Create a more recent environment to see if we can speedup training and/or inference
- [ ] Is it helpful to train for longer?
- [ ] Is it helpful to increase `max_seq_length_train`?
