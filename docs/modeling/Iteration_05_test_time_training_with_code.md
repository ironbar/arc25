# Iteration 5. Test-time training with code. Hindsight Experience Replay (HER)

_04-05-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## Goal

Explore if Hindsight Experience Replay (HER) can help to solve new tasks with code.

## Motivation

My idea is to pick a simple case that the model is not able to solve, for example: all vertical lines with all the colors. I first have to verify that the model is unable to solve it. Then check if using HER helps to solve the task.

## Development

### Design OOD tasks

I have prepared 3 simple but OOD tasks where the model is not able to find a solution despite being
sampled 256 times.

![alt text](res/1746350646030_image.png) ![alt text](res/1746350702470_image.png) ![alt text](res/1746350731746_image.png)

```bash
# vertical lines
mean_correct_pixels: 65.32%
max_correct_pixels: 88.89%

# squares
mean_correct_pixels: 64.55%
max_correct_pixels: 81.48%

# overlapping squares
mean_correct_pixels: 62.66%
max_correct_pixels: 93.00%
```

My guess is that the first two tasks are not solved due to requiring 9 draws (the model was trained with up to 5).
The last one might be difficult due to the overlapping squares.

¿Could test-time training allow the model to solve this tasks?

## Results

### HER allows to solve tasks with 25 squares

HER allows to solve tasks with 25 squares for a model that was trained with just up to 5 objects.

![best prediction evolution](res/1746622789551_image.png)

![distribution evolution](res/2025-05-07-15-01-52.png)

The model starts with a maximum accuracy of less than 50% and in 6 epochs is able to achieve perfect accuracy.

### Weaker models

If we use weaker models (models trained for a smaller number of iterations) they can still benefit from HER.
Their start point is worse, and they cannot solve new tasks as complex as the stronger models but nonetheless HER allows adaptation to novel tasks.

## Conclusion

## Next steps

## TODO

- [ ] Find a simple task that the model is not able to solve
- [ ] Does Hindsight experience replay helps to learn to do the task?
- [ ] Does RL helps?
- [ ] Is data augmentation helpful?
- [x] Colab pro: https://www.kaggle.com/docs/notebooks#increase-gpu-compute-with-colab-pro, Now I have 45 hours per week at Kaggle!