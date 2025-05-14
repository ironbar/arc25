# Iteration 9. Improve training script

_14-05-2025_

## Goal

Improve the training script so I can start working towards solving real ARC tasks with code.

## Motivation

I have seen that Hindsight Experience Replay (HER) allows to generalize to novel tasks. Next step is
to probe that it can solve real ARC tasks, not just toy tasks. But previously I have to make some updates
to the training script. That will allow me to iterate faster on the next steps.

## Development

## Results

## Conclusion

## Next steps

- Solve the training set, then the evaluation set, then the new tasks from ARC25.

## TODO

- [ ] Fix the problem with repeated calls to the train dataset generator
- [ ] Make the script work with accelerate
- [ ] Measure training speed vs input size
- [ ] Enable multi-task training, currently only trains on a single task
- [ ] Measure data sampling speed to verify is fast enough
- [ ] Add validation
