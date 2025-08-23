# Iteration 21. Fix bug with data

_23-08-2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

How good is the BARC induction model on the different ARC datasets?

## Motivation

I have discovered that I wasn't using the test samples when evaluating the BARC model. This make the problem harder in a way (because not all the training samples were given) and easier in another way (maybe the test samples are more difficult or cover some edge cases). On this iteration I need to stablish a good baseline so I can later check if test-time adaptation improves the scores.

I already know that data augmentation is helpful, so I will be using it by default on this iteration.

## Development

## Results

## Conclusion

## Next steps

## TODO

- [ ]
