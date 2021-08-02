# RL Classics
This repository is a playground in which I'm implementing a few classic function approximators for the ML portion of my 2021 "summer school" on simulation and reinforcement learning. These implementations are as short and straight forward as possible. Each file is self-contained, and running it will train a small linear or logistic model on a toy regression or classification problem. I also plan to implement these two models in `torch` to show what models look like in that framework, and demonstrate how it's nice not to have to calculate gradients ourselves.

## Linear Regression (numpy-regression.py)

Super dumb and basic logistic regressor implemented to demonstrate vanilla gradient descent on a linear model, with MSE cost.

## Logistic Regression (numpy-regression.py)

Super dumb and basic logistic regressor implemented to demonstrate vanilla gradient descent on a logistic model, with log probability / cross-entropy cost.
