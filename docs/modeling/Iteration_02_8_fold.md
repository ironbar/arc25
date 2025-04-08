# Iteration 2. Architects solution with 8 data splits

_04-04-2025_

## Goal

Can I improve the leaderboard score by doing 8 splits to the data instead of 4?

## Motivation

I believe that one of the reasons that my adaptation of the architects code is scoring higher than expected is that I'm using 4 data splits instead of the original 2 splits. On this iteration I want to see if 8 splits give even better results, if that is the case I will study how
to do an arbitrary number of splits (ideally equal to the number of tasks).

It might be enough to be the first team to break the barrier of 10% on the ARC-AGI-2 benchmark.

## Development

### ARC24 vs ARC25 parameters

On ARC24 there were 100 training tasks, they were split in two folds and they train for 4 epochs.
The effective batch size was 4, thus if I take data augmentation into account this means it trained for 400 steps (`100/2*8*4/4`). They warmup the learning rate for 100 steps, so 25% of the training time.

Now I want to use 8 folds, if I keep the number of epochs constant that means I will train for just 120 steps (`120/8*8*4/4`). On each fold there would be just 15 tasks.

Maybe the easiest solution is to use the `warmup_ratio` instead of `warmup_steps` and set it to a value such as 10%.

### Initial leaderboard score is very low

My first submission scores just 3.33 vs 7.08 that scored the submission with 4 folds. I have compared the code of the two submissions and I don't see anything wrong. What could be the cause of this drop in score?

- Maybe my first submission was just lucky
- The biggest change is that trainings are now shorter. Maybe I have to reduce the lora rank and/or increase the training epochs. 

After halving the lora rank to 32 and increasing the number of epochs from 4 to 6 the score improved to 6.67, still worse than the first submission but very close. However the submission time was very close to 12 hours. My guess is that there is an interplay between model training and inference, because the 2 additional epochs will likely take around 2000 seconds, and I saw an increase in submission time of around 10800 seconds.

### Training and inference times

Training is taking around 4 hours for 6 epochs, so that is around 40 minutes per epoch. Training time should be constant.
Inference time can change depending on the training, for 6 epochs of training it takes 4 hours to do 8 predictions per task.

## Results

I have not improved over my first submission of 7 but I was close (6.67). [Google Sheet with results](https://docs.google.com/spreadsheets/d/1NmmCZA7gPOyoBypwvpw_JhYdjcvqNFHibX_WahwTHIM/edit?gid=0#gid=0&range=A12)

A submission doing 8 predictions per task instead of 16 scored almost the same (6.25 vs 6.67).

## Conclusion

I have not improved over my first submission of 7 but I was close (6.67). I believe I should push this further and train on each task individually.

## Next steps

Create a notebook that allows to do training and inference on each task independently.

## TODO

- [ ] Try to understand low score of 3.33 -> Are there any errors? Compare with the best submission.
- [ ] Does it help to reduce the lora rank?
- [ ] n=1 with smaller min_prob?
- [ ] How long does it take to train on the private test set?