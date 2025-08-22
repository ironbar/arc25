# Iteration 20. Data augmentation with BARC

_21-08-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Does using data augmentation increases the diversity of the predictions and improves the pass@n metric?

## Motivation

On a previous [iteration with base models](Iteration_17_increase_search_diversity.md) I found that
data augmentation was not helpful. That result was weird, so I want to repeat the experiments with BARC.

## Development

### ARC-AGI-1 evaluation set is chosen

For these experiments I believe that evaluation set from ARC-AGI-1 has the greatest signal. The BARC
model was able to solve around 20% of the tasks. The scores on the training set are not trustable because
the model was trained on those or similar tasks, while the ARC-AGI-2 evaluation set is more difficult
and only 2 out of 120 tasks were solved.

### Experimental setup

The idea is to reuse all the data augmentation implemented on [iteration 17](Iteration_17_increase_search_diversity.md).
I will make predictions in batches of 8 or 16 predictions per task, and later I will aggregate all the predictions to
estimate the accuracy of the system. I will have to save the data augmentation configuration alongside each prediction
to be able to undo it when executing the code.

## Results

### Data augmentation improves the accuracy of the model by increasing the diversity of the predictions

![alt text](res/1755837928981_image.png)

The pass@n metric improves when using data augmentation. The difference is bigger when the number of predictions grows.
This could explain why my previous experiments with just 8 predictions per task did not show improvements.

| experiment        | n_preds | valid code | valid outputs | unique outputs | pixel similarity | correct grids | pass_rate | pass@n     |
|-------------------|---------|------------|---------------|----------------|------------------|---------------|-----------|------------|
| baseline          | 568     | 100.0%     | 75.9%         | 40.9%          | **57.1%**        | **3.0%**      | 1.96%     | 21.00%     |
| data augmentation | 584     | 100.0%     | **76.5%**     | **44.4%**      | 56.4%            | 2.9%          | **1.98%** | **24.50%** |

This is probably caused by having more diversity on the outputs, the metric that measure the unique outputs improves
from 40.9% to 44.4%.

### Trustability of the metrics

I have seen that we can only trust the metrics for a number of predictions around 1/4 of the total
of predictions run (at least for pass@n metric). There is a bias to underestimate the pass@n rate when
the number of predictions is small.

![alt text](res/1755840721821_image.png)

Thus when making comparisons between experiments we should try to have a similar number of predictions.

### Distribution of output tokens

![alt text](res/1755849933817_image.png)

I was using the `max_tokens=2048` from the previous iterations and it seems it is a good value.
The median output tokens seems to be around 400, and we can see that the datasets are sorted by inference speed as expected. We could probably be using 1024 output tokens without much impact on the results.
The important takeaway is that the current configuration is not hurting the accuracy of the model.

## Conclusion

## Next steps

## TODO

- [ ] Have a look at some of the solutions to verify they are legit implementations
- [x] Distribution of prediction length
