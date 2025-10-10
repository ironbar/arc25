# Solution Summary
<!--
https://www.kaggle.com/wiki/WinningModelDocumentationTemplate
https://www.kaggle.com/solution-write-up-documentation
--->

## Introduction

TODO: very brief description of ARC, what is intelligence and why it is important. Ability is not intelligence.
Requirements of a good intelligence test. Intelligence is all about adaptation to novelty.

## Vision

There are only two methods to adapt to novelty: search and learn.

All the top scoring solutions from ARC24 relied on learn: they used test-time training to adapt the
model to the new tasks.

In the other hand the solutions for the semi-private evaluation relied on search. o3 and other reasoning
models search the space of natural language programs to find solutions for novel tasks. Other methods
pioneered by Greenblatt searched the space of python programs.

Humans use both methods, when we approach a new task we try different approaches to try to solve it and
we learn from the failures. When trying subsequent approaches we do not repeat the mistakes, we try
new approaches that take into account the information obtained with the failing trials. So we search,
learn from our mistakes and start the cycle again until we eventually find the solution. For the
harder problems (like solving ARC) this cycle can take many years.

I believe that a system that will solve ARC will very likely combine search and learn as well. All my
work during the ARC25 challenge has gone in that direction.

## Brief story of my work for ARC25

TODO:
