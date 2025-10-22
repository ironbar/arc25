# Solution Summary
<!--
https://www.kaggle.com/wiki/WinningModelDocumentationTemplate
https://www.kaggle.com/solution-write-up-documentation
--->

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Solution Summary](#solution-summary)
  - [Abstract](#abstract)
  - [Introduction](#introduction)
  - [Vision](#vision)
    - [Four ways to arrive at that vision](#four-ways-to-arrive-at-that-vision)
      - [Search and learn](#search-and-learn)
      - [Combine the best approaches from ARC24: test-time training and program synthesis](#combine-the-best-approaches-from-arc24-test-time-training-and-program-synthesis)
      - [Imitate how humans solve ARC](#imitate-how-humans-solve-arc)
        - [How humans solve ARC](#how-humans-solve-arc)
        - [How AI might solve ARC](#how-ai-might-solve-arc)
      - [Frame ARC as a game and solve it with RL](#frame-arc-as-a-game-and-solve-it-with-rl)
    - [Why it will beat the other approaches](#why-it-will-beat-the-other-approaches)
      - [Transduction and test-time training](#transduction-and-test-time-training)
      - [Natural language program search (o3)](#natural-language-program-search-o3)
      - [Evolutionary program search](#evolutionary-program-search)
  - [Brief story of my work for ARC25](#brief-story-of-my-work-for-arc25)
  - [Content](#content)
    - [1. How does test-time training compares against o3?](#1-how-does-test-time-training-compares-against-o3)
    - [2. Does hindsight relabeling works for program synthesis on toy tasks?](#2-does-hindsight-relabeling-works-for-program-synthesis-on-toy-tasks)
    - [3. Does hindsight relabeling works for program synthesis on ARC tasks?](#3-does-hindsight-relabeling-works-for-program-synthesis-on-arc-tasks)
    - [4. Can we get a stronger base model with reinforcement learning?](#4-can-we-get-a-stronger-base-model-with-reinforcement-learning)
    - [5. Can we improve the search accuracy by doing prediction refinement?](#5-can-we-improve-the-search-accuracy-by-doing-prediction-refinement)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Abstract

## Introduction

TODO: very brief description of ARC, what is intelligence and why it is important. Ability is not intelligence.
Requirements of a good intelligence test. Intelligence is all about adaptation to novelty.

## Vision

**ARC will be solved first by deep-learning-guided program synthesis that searches program space and adapts at test time with test-time training via hindsight relabeling, in a tight search-and-learn loop.**

### Four ways to arrive at that vision

TODO: read and rewrite this paths

#### Search and learn

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

<center><img src="modeling/res/1752753996905_arc25.png" width="50%"></center>

#### Combine the best approaches from ARC24: test-time training and program synthesis

Last year's competition showed that test-time training allowed the models to adapt to the novel tasks. At the same time in the semi-private dataset we saw that frontier models could generate code to solve more than half of the tasks.

Using code is a more promising approach because:

1. It is verifiable
2. Enables to iteratively refine the solution by comparing the outputs with the ground truth. I would argue that this is similar to reasoning.

My hypothesis is that we can use [hindsight experience replay (HER)](https://arxiv.org/abs/1707.01495) at test time to update the beliefs of the model and find the right solution more efficiently. Instead of sampling thousands of programs, sample a few and learn from the mistakes. **That is the way to combine induction and test-time training.**

We can treat the failed code attempts that run as new tasks, and train the model on those tasks. Those tasks will be in the neighborhood of the task that we want to solve.

We already know that HER enables faster learning, specially in very sparse reward environments.

![](res/2025-03-25-16-38-36.png)

Additionally we could define a continuous metric such as the number of correct pixels and use it with reinforcement learning to modify the model towards solutions
that score higher.

#### Imitate how humans solve ARC

##### How humans solve ARC

![](res/how-humans-solve-arc.png)

When humans try to solve ARC tasks we draw some hypothesis and test it in our heads, if it is not correct we update our beliefs and refine the hypothesis. What modules are needed to do this process?

- **Policy.** What action do I have to do to achieve the goal? Learned with hindsight
- **World model.** What happens if do this action? Learned with past experiences
- **Judgment.** Is the solution correct? Learned with human feedback or by comparison
- **Learning.** In difficult problems we are able to learn from our errors and modify our initial beliefs about the problem.

Reasoning is an iterative process, as shown in the loop diagram in the image.

##### How AI might solve ARC

Focusing on efficiency the best configuration for ARC might be the following:

![](res/how-ai-might-solve-arc.png)

- **Policy**: a Large Reasoning Model.
- **World model**: python interpreter
- **Judgment**: metric function
- **Learning**: reinforcement learning and hindsight experience replay

That way we only have to learn the policy and parametrize the learning, all the other modules are guaranteed to work perfectly.


#### Frame ARC as a game and solve it with RL

The idea is to frame ARC as a reinforcement learning problem. The system is given a new task and it needs to learn it as efficiently as possible. It is like playing a game, but instead of hitting the buttons it has to write code to play.
The code generates an output that is evaluated against the ground truth and returns an score.

Finding the right program is equivalent to finding the right trajectory to solve a game. Instead of actions we write code, but the problem is exactly the same. When we want to solve a new task in ARC is the same as wanting to solve a new game. We can frame the problem as a Reinforcement learning game, with a very sparse reward.

TODO: standard RL does not work well with very sparse rewards, HR is needed.

### Why it will beat the other approaches

ARC can be solved (and will be solved) with many different approaches, but in this section I will
argue why search and learn will be the first approach to solve it.

TODO: why code is better than transduction and natural language

#### Transduction and test-time training

TODO: programming is better suited for ARC-AGI-2 tasks with multiple interacting rules
TODO: dominant approach on ARC24 and probably ARC25

#### Natural language program search (o3)

TODO: python interpreter is perfect, models fail, specially out of distribution

#### Evolutionary program search

TODO: a frozen model won't be able to generalize when the generalization jump is big

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
8. Experimented with prediction refinement, but BARC model does not seem to have that capability
   that allows techniques like AlphaEvolve with frontier models.
9. I have tried to make search and learn more hardware efficient by grouping the tasks (instead
    of training on each task independently) but I wasn't able to find a good configuration. Each
    evaluation takes 3 days on a single GPU, so iteration was very slow.

Now I'm trying to:

- Solve RL training collapse so I can train for longer. Hoping that training on huge datasets helps.
- Trying to make the model learn to refine its predictions using RL, but maybe I should start thinking
  on a fresh system for ARC26. 

In essence the BARC model is not strong enough and doesn't know how to refine its predictions.

## Content

### 1. How does test-time training compares against o3?

At the start of ARC25 challenge I was curious to see how well test-time training
compared against o3. A custom version of o3 was presented in December 2024 and reported to have solved 87.5% of the semi-private test set of ARC-AGI-1. However with the release
of ARC-AGI-2 o3 was solving less than 5% of the semi-private test set. It was not
the exact same version of o3, but the change was so dramatic.

To my surprise I was able to score 11.94 on the leaderboard, doubling the score of o3
and being the first team to score above 10% in the challenge.


### 2. Does hindsight relabeling works for program synthesis on toy tasks?

### 3. Does hindsight relabeling works for program synthesis on ARC tasks?

### 4. Can we get a stronger base model with reinforcement learning?

### 5. Can we improve the search accuracy by doing prediction refinement?

## Acknowledgements

