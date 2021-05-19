import time
import matplotlib.pyplot as plt
import numpy as np
from GPGO import GP
from GPGO import RBF
from GPGO import BayesianOptimization, MultiObjectiveBO
from GPGO.GaussianProcess import log_gp
from GPGO.GaussianProcess import generate_grid
import logging
import deap


def f(x):
    x = x[0]

    def f1(x):
        if x <= 1:
            return -x
        if 1 < x <= 3:
            return x - 2
        if 3 < x <= 4:
            return 4 - x
        if x > 4:
            return x - 4

    def f2(x):
        return (x - 5) ** 2 - 10

    return f1(x), f2(x)


b_low = 0
b_hi = 1
n = 30
X = np.random.uniform(0, 1, n)
Y = []
X = []
for i in range(100):
    X.append(np.random.uniform(0, 1, n))
    Y.append(deap.benchmarks.zdt3(X[i]))
X = np.array(X)
Y = np.array(Y)
print(X)

t = MultiObjectiveBO(X, Y, 2, func=None, settings={
    "ac_type": "EI",
    "boundaries": np.array([[0, 1] for i in range(30)]),
    "epsilon": 0.1,
    "iteration": 1,
    "func": "ac",
    "batch": 1,
    "minimization": True})

op = t.suggest_location()
print(op)
#plt.scatter(op.NSGAII["pareto"][:, 0], op.NSGAII["pareto"][:, 1])
#plt.show()