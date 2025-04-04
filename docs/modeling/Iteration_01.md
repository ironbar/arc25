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

### Speeding-up inference

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

## Results

## Conclusion

## Next steps

- Everyday we are allowed to make submissions with a compute cost of 50$. Over 7 months that will be around 10k$. That is a lot of money, maybe I should use the private test set as my development set and use the evaluation set as the test set (to decide the final submission).

## TODO

- [ ]
