# Iteration 18. Refine Solution

_31-07-2025_

## Goal

Experiment with solution refinement with base models. Does it improve over simply sampling the model
multiple times?

## Motivation

I'm still exploring the search strategy to find solutions for ARC. In [SOAR](https://icml.cc/virtual/2025/poster/43499) and on Ryan Greenblatt
solution the model was able to refine the functions and that was beneficial. I want to verify that
I can also do it.

## Development

To be able to refine a function, I believe we should give the model the output grids and the scores.
That information is useful to modify the initial function.

I could give all the information on the user message, or simulate a multi-turn conversation with the agent.
I believe they should give similar results but I need to check it.

The goal of the experiment is to check if doing function refinement leads to better accuracy
than simply sampling the model.

## Results

## Conclusion

## Next steps

## TODO

- [ ]
