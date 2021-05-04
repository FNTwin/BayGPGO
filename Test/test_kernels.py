import time
import matplotlib.pyplot as plt
import numpy as np
from GPGO import GP
from GPGO import RBF as RBF1
from GPGO import BayesianOptimization
from GPGO.GaussianProcess import log_gp
from GPGO.GaussianProcess import generate_grid
import logging
from GPGO.GaussianProcess.Kernel import Matern, Bias

x = np.random.uniform(-3, 3, 7)[:, None]


def f(X):
    # return -(1.4-3*X)*np.sin(18*X)
    # return X**0
    return np.sin(X)
    # return (6 * X - 2) ** 2 * np.sin(12 * X - 4) - X
    # return X + np.sin(X)*10


def noise(x, alpha=1):
    return f(x) + np.random.randn(*x.shape) * alpha


y = f(x)

a = Matern()
print(a.gethyper())
gp = GP(x, y, kernel=Matern(gradient=False), normalize_y=True)
gp.fit()
# gp.plot(np.linspace(-4,4,100)[:,None])
gp.optimize(n_restarts=10, optimizer="L-BFGS-B")
# gp.plot(np.linspace(-4,4,100)[:,None])
# gp.get_kernel().plot()
print(gp.get_kernel().gethyper())
# b=(RBF()+RBF()+a)*a

# print(RBF().__dict__)


dim_test = 2
dim_out = 1
n_train_p = 7
# X = np.random.uniform(-2.5,2.5, (25, 2))
# X = generate_grid(dim_test, 5, [[-4, 4] for i in range(dim_test)])
X = np.random.uniform(-5, 5, (10, 2))


def f(o):
    x = o[:, 0]
    y = o[:, 1]
    return ((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2)[:, None]


# Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
# Z = ((X[:, 1] * X[:, 0]) / np.exp((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
# Z=np.sin((X[:, 1] ** 2 + X[:, 0] ** 2))[:,None]
gp = GP(X, f(X), kernel=RBF1())
gp.fit()
plot = generate_grid(dim_test, 50, [[-5, 5] for i in range(dim_test)])


# gp.plot(plot)

def test_Hartmann_6D():
    dim = 6
    points = 10
    x = np.random.uniform(0, 1, (10, 6))

    def hartmann_6D(x):
        alpha = np.array([[1.], [1.2], [3.], [3.2]])

        A = np.array([[10, 3, 17, 3.50, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])

        P = 10 ** -4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                                 [2329, 4135, 8307, 3736, 1004, 9991],
                                 [2348, 1451, 3522, 2883, 3047, 6650],
                                 [4047, 8828, 8732, 5743, 1091, 381]])

        def comp(i):
            tot = 0
            for j in range(6):
                 tot += A[i][j] * (x.T[j] - P[i][j]) ** 2
            return np.exp(-tot)

        f = 0
        for i in range(4):
            f += -(alpha[i] * comp(i))

        return f[:, None]

    y = hartmann_6D(x)

    gp = GP(x, y, RBF1(gradient=False))
    gp.fit()

    settings = {"type": "DIRECT",
                "ac_type": "UCB",
                "n_search": 10,
                "boundaries": [[0, 1] for i in range(6)],
                "epsilon": 0.1,
                "iteration": 50,
                "minimization": True,
                "optimization": True,
                "n_restart": 30,
                "sampling": np.random.uniform}

    BayOpt = BayesianOptimization(x, y, settings, gp, hartmann_6D)

    n_p = 10

    best = BayOpt.run()

    print("Number of points sampled in an iteration: ", n_p ** dim)
    print("bay:", best)
test_Hartmann_6D()

sockeye_data = np.reshape([2986,9,
3424,12.39,
1631,4.5,
784,2.56,
9671,32.62,
2519,8.19,
1520,4.51,
6418,15.21,
10857,35.05,
15044,36.85,
10287,25.68,
16525,52.75,
19172,19.52,
17527,40.98,
11424,26.67,
24043,52.6,
10244,21.62,
30983,56.05,
12037,29.31,
25098,45.4,
11362,18.88,
24375,19.14,
18281,33.77,
14192,20.44,
7527,21.66,
6061,18.22,
15536,42.9,
18080,46.09,
17354,38.82,
17301,42.22,
11486,21.96,
20120,45.05,
10700,13.7,
12867,27.71,], (34,2))

import GPy
import numpy as np
import pandas as pd

import scipy as sp
def weinland(t, c, tau=4):
    return (1 + tau * t / c) * np.clip(1 - t / c, 0, np.inf) ** tau

def angular_distance(x, y, c):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return (x - y + c) % (c * 2) - c

C = np.pi
x = np.linspace(0, C)
plt.figure(figsize=(16, 9))
for tau in range(4, 10):
    plt.plot(x, weinland(x, C, tau), label=f"tau={tau}")
plt.legend()
plt.ylabel("K(x, y)")
plt.xlabel("dist");
plt.show()
angles = np.linspace(0, 2 * np.pi,100)
X=np.random.uniform(0, np.pi * 2, size=5)
Y= X[:,None]**2 * np.sin(X[:,None])
observed = dict(x=X.ravel(), y=Y.ravel())

kernel = GPy.kern.RBF(input_dim=1, variance = 1., lengthscale= 1.)
m = GPy.models.GPRegression(observed["x"][:,None],observed["y"][:,None], kernel)
m.optimize_restarts(num_restarts = 20)
m.plot()
plt.show()
gp = GP(observed["x"][:,None],observed["y"][:,None], kernel=RBF1(), normalize_y=True)
gp.fit()
gp.set_boundary([[1e-3,50],[1e-3,50],[1e-3,50]])
gp.optimize(n_restarts=40)
print(gp.get_kernel().gethyper())

y_pred=gp.predict(angles[:,None])
plt.polar(angles, y_pred[0].ravel(), color="black")
plt.scatter(observed["x"], observed["y"], color="red", marker="x", label="observations")

plt.fill_between(
        angles,
        (y_pred[0] - y_pred[1]** 0.5).ravel(),
        (y_pred[0] + y_pred[1] ** 0.5).ravel(),
        color="gray",
        alpha=0.5,
        label=r"$\mu\pm\sigma$",
    )


plt.show()



