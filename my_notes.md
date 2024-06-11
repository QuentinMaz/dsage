# About this fork
This fork aims to summarise my studies and use of the original repository.
My goals is mostly to borrow the Maze experiment (including the agent under test) and the proposed occupancy prediction model.
# My Installation
Create a conda environment (Python 3.8.11) and install all the dependancies:
```
conda create -n dsage python=3.8.11
conda activate dsage
pip install -r requirements.txt
```
# Code Notes
Personal notes of the code for borrowing the Maze experiments.
The objective is to understand:
- How to load the agent under test.
- How to create inputs (i.e., valid random Maze).
- How to mutate inputs (i.e., mutate generated Mazes).
- Possibly, how to train an occupancy grid predictor.
## Maze Env
The environment is based on PAIRED prior work.
Inputs (or levels) are generated randomly.
Then, the shortest path between??? creates the start and goal positions.
From the detailed sections C, an input is evolved by randomly changing 10 values.
Directions are randomly set when resetting the env (see place_agent_at_pos from Maze class).
I will use the levels as inputs and assume that each maze has then the same start and goal positions.
Finally, regarding the direction, I can either a particular one or keep random ones since it is very likely that inputs will be evaluated multiple times.
To that regard, the only question is the reduction of the failure results... failure probability?
## Maze Agent
