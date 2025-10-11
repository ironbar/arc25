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

1. Baseline with test-time training. Since o3 was solving less than 5% of the test tasks,
   I wanted to see what was the accuracy of the last year's most successfull approach.
   To my surprise I was able to score 11.84, being the first team to score above 10 in the challenge.
2. Then I moved and started to explore if an LLM generating code could learn from its failing attempts
   and generalize outside its training distribution. For that I designed a toy environment where
   the model had to learn to draw. I probed that hindsight relabelling was able to adapt the model
   to work with much more complex drawings than the ones seen during training.
3. Next step was to see if the same could be applied for ARC tasks, much more difficult than the
   toy problem of drawing. My initial view was to create new tasks to teach how to use the primitive
   functions of a custom DSL created for ARC. The problem was that I underestimated the difficulty
   of creating a big number of diverse training tasks so the model could learn to use the primitive
   functions effectively. Our current deep learning methods need a lot of data, and data needs to be
   very diverse. My training tasks were very few and not very diverse.
4. Then I tried to use public models, and let them use the DSL by describing it on the prompt. But it did not work well, the models were not able to explore the space effectively, there were a lot of repeated solutions.
5. Thus I decided to switch and use the BARC induction model. That model had been trained with a
   lot of ARC tasks and was able to use a DSL. Maybe the DSL was not complete or maybe the model
   was not strong enough, but I believed I could probe or discard my ideas with it. On a first
   step I validated that the model could produce reasonable good results with a reasonable number of
   predictions (<= 512), because in the BARC paper I believe they use 20k predictions.
6. Validated that using search and learn the BARC induction model improves its accuracy from 22% to 27%.
   The improvement is not dramatic but it solves tasks that won't be solvable using simple predictions.
7. Start working with RL to try to improve the solve rate of the model, that way I won't be needing
   too many predictions to solve each task. I have some early results that show it's a good direction,
   but trainings collapse.

Now I'm trying to:

- Solve RL training collapse so I can train for longer
- Try to make the search and learn script more hardware efficient
- Check if solution refinement is possible with the BARC model
