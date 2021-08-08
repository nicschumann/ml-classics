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

X = 200. * torch.rand(N, 1).type(torch.FloatTensor) - 100.
Y = f_true(X) + torch.normal(torch.zeros(N, 1), error_spread).type(torch.FloatTensor)


# Model

alpha = 0.00005

# model function

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.w1 = torch.tensor([0.0], requires_grad=True)
        self.w2 = torch.tensor([0.0], requires_grad=True)

    def forward(self, x):
        return self.w1 * x + self.w2

f_model = Model()
optimizer = SGD([f_model.w1, f_model.w2], lr=alpha)


# Monitoring Code

fig, (ax1, ax2) = plt.subplots(1, 2)

def plot(X, Y, costs):
    ax1.cla()
    ax1.set_title('Data')
    ax1.scatter(X, Y, color="blue")

    inp = torch.tensor([[-100.], [100.]])
    out = f_model(inp)

    ax1.plot(inp, f_true(inp), color="lightblue")
    ax1.plot(inp.detach().numpy(), out.detach().numpy(), color="red")

    ax2.set_title('Costs')
    ax2.plot(list(range(len(costs))), costs, color="blue")

    plt.pause(1.0)


# Gradient Descent

costs = []

plot(X, Y, costs)

for i in range(1000):

    Y_pred = f_model(X)
    loss = (Y - Y_pred).pow(2).mean()
    costs.append(loss.detach().numpy())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plot(X, Y, costs)


plt.show()
