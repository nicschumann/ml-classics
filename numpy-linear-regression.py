import numpy as np
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

X = np.random.uniform(low=-100, high=100, size=(N,))
Y = f_true(X) + np.random.normal(loc=0.0, scale=error_spread, size=(N,))



# Model

w0 = 0 #np.random.normal()
w1 = 0 #np.random.normal()
alpha = 0.0001

# model function
def f_model(X):
    return w0 * X + w1

# cost of model
def cost(X):
    return 1 / N * np.sum( np.power((Y - f_model(X)), 2))

# gradient of model w.r.t w0
def d_cost_d_w0(X):
    return -2 / N * np.sum(X * (Y - f_model(X)))

# gradient of model w.r.t w1
def d_cost_d_w1(X):
    return -2 / N * np.sum(Y - f_model(X))



# Monitoring Code

fig, (ax1, ax2) = plt.subplots(1, 2)

def plot(X, Y, costs):
    ax1.cla()
    ax1.set_title('Data')
    ax1.scatter(X, Y, color="blue")
    ax1.plot([-100, 100], [f_true(-100), f_true(100)], color="lightblue")
    ax1.plot([-100, 100], [f_model(-100), f_model(100)], color="red")

    ax2.set_title('Costs')
    ax2.plot(list(range(len(costs))), costs, color="blue")

    plt.pause(1.0)


# Gradient Descent

costs = []

plot(X, Y, costs)

for i in range(10):
    c = cost(X)

    costs.append(c)

    w0 = w0 - alpha * d_cost_d_w0(X)
    w1 = w1 - alpha * d_cost_d_w1(X)

    plot(X, Y, costs)


plt.show()
