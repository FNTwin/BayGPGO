import time
import matplotlib.pyplot as plt
import numpy as np
from GPGO import GP
from GPGO import RBF
from GPGO import BayesianOptimization
from GPGO.GaussianProcess import log_gp
from GPGO.GaussianProcess import generate_grid
import logging
from GPGO.GaussianProcess.Kernel import Matern, Bias

x=np.random.uniform(-3,3,7)[:,None]
def f(X):
    #return -(1.4-3*X)*np.sin(18*X)
    #return X**0
    return np.sin(X)
    #return (6 * X - 2) ** 2 * np.sin(12 * X - 4) - X
    #return X + np.sin(X)*10

def noise(x, alpha=1):
    return f(x) + np.random.randn(*x.shape) * alpha

y=f(x)

a=Matern()
print(a.gethyper())
gp = GP(x, y, kernel=Bias(gradient=False), normalize_y=True)
gp.fit()
gp.plot(np.linspace(-4,4,100)[:,None])
gp.optimize(n_restarts=20)
gp.plot(np.linspace(-4,4,100)[:,None])
gp.get_kernel().plot()
print(gp.get_kernel().gethyper())
#b=(RBF()+RBF()+a)*a

#print(RBF().__dict__)


dim_test = 2
dim_out = 1
n_train_p = 7
# X = np.random.uniform(-2.5,2.5, (25, 2))
X = generate_grid(dim_test, 5, [[-2.4, 2.4] for i in range(dim_test)])
# Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
Z = ((X[:, 1] * X[:, 0]) / np.exp((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
# Z=np.sin((X[:, 1] ** 2 + X[:, 0] ** 2))[:,None]
gp = GP(X, Z, kernel=Matern())
gp.fit()
plot = generate_grid(dim_test, 50, [[-2.5, 2.5] for i in range(dim_test)])


# pred = gp.predict(plot)
gp.plot(plot)


gp.optimize(n_restarts=10)
pred = gp.predict(plot)

gp.predict(plot)



