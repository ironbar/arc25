# Iteration 23. All in with test-time training with BARC induction model

_02/09/2025_

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.

<details>
  <summary>Click to expand/collapse this section</summary>
</details>
--->

## Goal

Create an efficient implementation of test-time training with BARC that tries to solve
each task independently.

## Motivation

On the previous [iteration](Iteration_22_ttt_BARC.md) I have seen that TTT is able to improve
the solving rate of the BARC induction model. That experiment was done using all the
tasks at once. I already know from the previous competition that is better to solve each task independently,
I believe that creates a cleaner gradient signal.

Also I know from my [toy experiments](Iteration_08_improve_HER.md) that multiple iterations of search and learn are needed
to solve tasks that are far from the training distribution. Sometimes requiring in the
order of tens of epochs.

Thus in this iteration I want to implement an efficient way to do search and learn
in multiple epochs. If the implementation is successful it will very likely be part
of my solution for the 2025 challenge.

## Development

### Implementation ideas

- Inference with VLLM is very efficient, and I can use different LoRAs which is convenient for test-time training.
- trl could be used for training, although I don't know if it is the best option.
- I believe unsloth is integrated with VLLM, which will make inference as fast and maybe is the
  best way to do inference and training in the same process. Otherwise I would have to have a
  training service, an inference service and a master service that redirects the traffic between
  the two.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
