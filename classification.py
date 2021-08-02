import numpy as np
import matplotlib.pyplot as plt

# This is a simple introduction to gradient-based function approximation
# of the kind used in deep learning. I'll be focusing on supervised, classification
# problems in this examples, as, in my view, these are the simplest ones.

# In this walkthrough, we'll ta

# ground truth parameters
m_true = 0.2
b_true = 0.0

# this is the decision boundary that separates points in class 1
# from points in class 2, for any 2D datapoint p = (x,y), p is in Class 1
# iff y <= f_true(x). Otherwise p is in Class 2.
# For reference:
f_true = lambda x: m_true * x + b_true



# Now, let's generate a candidate dataset. This is just a bunch of random datapoints
# that follow this rule.
n_train_samples = 10
n_test_samples = 20


X_train = np.random.uniform(low=-10., high=10., size=(n_train_samples, 2))
Y_train = (f_true(X_train[:, 0]) < X_train[:, 1]).astype('float32')


X_test = np.random.uniform(low=-10., high=10., size=(n_test_samples, 2))
Y_test = (f_true(X_test[:, 0]) < X_test[:, 1]).astype('float32')


# Model

w0 = 0
w1 = 0
w2 = 0
alpha = 0.01

def f_model(X):
    a = w0 * X[:, 0] + w1 * X[:, 1] + w2
    return 1.0 / (1 + np.exp(-a))

def cost():
    cost = np.zeros(Y_train.shape)
    Y_pred = f_model(X_train)
    cost[Y_train == 1.] = -np.log(Y_pred)[Y_train == 1.]
    cost[Y_train == 0.] = -np.log(1.0 - Y_pred)[Y_train == 0.]

    return (1.0/n_train_samples) * np.sum(cost)

def d_cost_d_w0():
    return (2.0/n_train_samples) * np.sum(X_train[:, 0] * (f_model(X_train) - Y_train))

def d_cost_d_w1():
    return (2.0/n_train_samples) * np.sum(X_train[:, 1] * (f_model(X_train) - Y_train))

def d_cost_d_w2():
    return (2.0/n_train_samples) * np.sum(f_model(X_train) - Y_train)


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
costs = []

def plot():
    colors = list(map(lambda x: 'orange' if x == 1.0 else 'blue', Y_train.tolist()))
    ax1.set_title('Train Points')
    ax1.scatter(X_train[:, 0], X_train[:, 1], edgecolors=colors, facecolors=colors)

    colors = list(map(lambda x: 'orange' if x == 1.0 else 'blue', Y_test.tolist()))
    pred_colors = list(map(lambda x: 'orange' if x > 0.5 else 'blue', f_model(X_test).tolist()))
    ax2.set_title('Test Points')
    ax2.scatter(X_test[:, 0], X_test[:, 1], edgecolors=colors, facecolors=pred_colors)

    ax3.set_title('Loss')
    ax3.plot(range(len(costs)), costs, color='blue')

    plt.pause(1)


plot()

for i in range(1000):

    # print(f_model(X_train))
    #
    # print(d_cost_d_w0())
    c = cost()
    costs.append(c)
    print(c)

    w0 = w0 - alpha * d_cost_d_w0()
    w1 = w1 - alpha * d_cost_d_w1()
    w2 = w2 - alpha * d_cost_d_w2()

    plot()

plt.show()
