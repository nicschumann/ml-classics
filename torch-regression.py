import numpy as np
import torch
from torch.optim import SGD

import matplotlib.pyplot as plt

# This is a simple introduction to gradient-based function approximation
# of the kind used in deep learning. I'll be focusing on supervised, classification
# problems in this examples, as, in my view, these are the simplest ones.

# In this walkthrough, we'll ta

# ground truth parameters
w0_true = 0.5
w1_true = 1.0
error_spread = 10.0

# this is the decision boundary that separates points in class 1
# from points in class 2, for any 2D datapoint p = (x,y), p is in Class 1
# iff y <= f_true(x). Otherwise p is in Class 2.
# For reference:
f_true = lambda X: w0_true * X + w1_true



# Now, let's generate a candidate dataset. This is just a bunch of random datapoints
# that follow this rule.
N = 50

X = torch.tensor(np.random.uniform(low=-100, high=100, size=(N,)))
Y = torch.tensor(f_true(X) + np.random.normal(loc=0.0, scale=error_spread, size=(N,)))



# Model

w0 = torch.tensor([0.], requires_grad=True) #np.random.normal()
w1 = torch.tensor([0.], requires_grad=True) #np.random.normal()
alpha = 0.005

# model function
def f_model(X, w0, w1):
    return w0 * X + w1

# cost of model
def cost(w0, w1):
    return (1.0 / N) * torch.sum( torch.pow((Y - f_model(X, w0, w1)), 2))

optimizer = SGD([w0, w1], lr=alpha)


# Monitoring Code

fig, (ax1, ax2) = plt.subplots(1, 2)

def plot(X, Y, costs):
    ax1.cla()
    ax1.set_title('Data')
    ax1.scatter(X, Y, color="blue")
    ax1.plot([-100, 100], [f_true(-100), f_true(100)], color="lightblue")

    inp = torch.tensor([-100, 100])
    out = f_model(inp, w0, w1)

    print(inp)
    print([w0, w1])
    print(out)

    ax1.plot(inp.detach().numpy(), out.detach().numpy(), color="red")

    ax2.set_title('Costs')
    ax2.plot(list(range(len(costs))), costs, color="blue")

    plt.pause(1.0)


# Gradient Descent

costs = []

plot(X, Y, costs)

for i in range(10):


    optimizer.zero_grad()
    c = cost(w1, w0)
    costs.append(c.detach().numpy())
    c.backward()
    optimizer.step()

    plot(X, Y, costs)


plt.show()
