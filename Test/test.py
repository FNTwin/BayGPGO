import time
import matplotlib.pyplot as plt
import numpy as np
from GPGO import GP
from GPGO import RBF
from GPGO import BayesianOptimization
from GPGO.GaussianProcess import log_gp
from GPGO.GaussianProcess import generate_grid
import logging

def test_GP_1D(optimize=True):
    #x =  np.arange(-3, 5, 1)[:, None]
    #x=np.array([0.1,0.12,0.143,0.3,0.5,0.75,0.67,0.9,0.92,1.1])[:,None]
    x=np.random.uniform(-3,3,2)[:,None]

    def f(X):
        #return -(1.4-3*X)*np.sin(18*X)
        #return X**0
        return np.sin(X)
        #return (6 * X - 2) ** 2 * np.sin(12 * X - 4) - X
        #return X + np.sin(X)*10



    def noise(x, alpha=1):
        return f(x) + np.random.randn(*x.shape) * alpha

    #y = noise(x, alpha=0)
    y=f(x)
    print(y)


    #gp = GP(x*10000, y*1000, kernel=RBF(sigma_l=0.2, l= 1, noise= 1e-3, gradient=False), normalize_y=True)
    gp = GP(x, y, kernel=RBF(gradient=True), normalize_y=True)
    gp.fit()

    plot = np.linspace(-3,3, 1000)
    gp.set_boundary([[1e-5,3]])

    #pred_old, var_old = gp.predict(plot[:, None])

    #gp.plot(plot[:, None])
    gp.log_marginal_likelihood()
    print("Old marg likelihood :", gp.get_marg(), "\n Hyperparameters: ",
          gp.get_kernel().gethyper())
    if optimize:
        """new = gp.grid_search_optimization(constrains=[[1, 30], [1, 30],[0.00001,1]],
                                          n_points=100,
                                          function=np.linspace)"""


        gp.optimize(n_restarts=10, optimizer="L-BFGS-B", verbose=False)
        #gp.optimize_grid(n_points=50)


        #optimized.fit()
        #pred, var = gp.predict(plot[:, None])

        #plt.plot(plot[:,None],f(plot))
        gp.predict(plot[:, None])
        #plt.scatter(x,y,marker="x",color="red")
        gp.log_marginal_likelihood()
        log_gp(gp)
        print(gp.get_kernel().gethyper())

        #print(gp)


def test_GP_2D(optimize=True, function=np.linspace):
    dim_test = 2
    dim_out = 1
    n_train_p = 7
    #X = np.random.uniform(-2.5,2.5, (25, 2))
    X = generate_grid(dim_test, 5, [[-2.4,2.4] for i in range(dim_test)])
    #Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    Z = ((X[:, 1] * X[:, 0])/ np.exp((X[:, 1] ** 2 + X[:, 0] ** 2))) [:, None]
    #Z=np.sin((X[:, 1] ** 2 + X[:, 0] ** 2))[:,None]
    gp = GP(X, Z, kernel=RBF())
    gp.fit()
    plot = generate_grid(dim_test, 50, [[-2.5,2.5] for i in range(dim_test)])
    #print(plot.shape)

    #pred = gp.predict(plot)
    gp.plot(plot)
    # gp.static_compute_marg()
    #print("Old marg likelihood :", gp.get_marg(),
          #"\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        #gp.set_boundary([[1e-4,1]])
        gp.optimize(n_restarts=10)
        pred = gp.predict(plot)
        #print(pred)
        gp.predict(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())


def test_GP_4D(optimize=False):
    x = generate_grid(4, 3, [[-2, 2] for i in range(4)], np.random.uniform)


    def f(x):
        return x[:, 1] ** 2 - x[:, 3] * x[:, 0]

    y = f(x)[:, None]
    plot = generate_grid(4, 5, [[-2, 2] for i in range(4)], np.linspace)
    gp = GP(x, y,  kernel=RBF(sigma_l=2, l=2))
    gp.fit()
    mean, var = gp.predict(plot)
    print("Old marg likelihood :", gp.get_marg(),
          "\n Hyperparameters: ", gp.get_kernel().gethyper())

    if optimize:
        gp.optimize_grid(constrains=[[1, 3], [2, 100], [0, 30]], n_points=100, function=np.random.uniform)
        mean, var = gp.predict(plot)
        print("New marg likelihood :", gp.get_marg(),
              "\n Hyperparameters: ", gp.get_kernel().gethyper())

    return mean, var


def test_minimization_2D():
    dim_test = 2
    dim_out = 1
    n_train_p = 3
    X = np.array([[-4.1,9.3]])
    boundaries = [[-5, 10], [0, 15]]

    def f(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (1 * (x2 - (5.1 / (4 * np.pi ** 2)) * x1 ** 2 + 5 / np.pi *
                     x1 - 6) ** 2 + 10 * (1 - (1 / (8 * np.pi))) * np.cos(x1) + 10)

    Z = f(X)[:, None]
    gp = GP( X, Z, RBF(gradient=False), normalize_y=False)
    gp.fit()
    settings = {"type": "BFGS",
                "ac_type": "EI",
                "n_search": 3,
                "boundaries": boundaries,
                "epsilon": 0.01,
                "iteration": 10,
                "minimization": True,
                "optimization": True,
                "n_restart": 5,
                "sampling": np.linspace}

    BayOpt = BayesianOptimization(X,Z, settings, gp, f)

    best=BayOpt.suggest_location()

    print("bay:", best)

    # plot = generate_grid(dim_test, 30, [[-5, 5] for i in range(dim_test)])
    """plot = generate_grid(dim_test, 30, boundaries, np.linspace)
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Z, color='red', marker="x")
    ax.scatter(plot[:, 0], plot[:, 1], f(plot))
    ax.scatter(best[0][0], best[0][1], best[1], color="red")
    ax.scatter(gp.get_X()[:, 0], gp.get_X()[:, 1], gp.get_Y(), marker="x", color="black")
    plt.show()"""


def test_minimization_1D():


    X = np.random.uniform(-10,20,3)[:,None]


    def f(X):
        #return -(1.4 - 3 * X) * np.sin(18 * X)
        #return (6* X - 2)**2 * np.sin (12 * X - 4)
        return X**1
        #return 4 * 100 * ((.9 / X) ** 12 - (.9 / X) ** 6)

    def noise(X):
        return f(X) + np.random.randn(*X.shape)*0.4

    Z = f(X)

    gp = GP(X, Z, RBF(gradient=False), normalize_y=True)
    #gp.set_boundary([[1e-4,0.5]])
    settings={"type":"DIRECT",
              "ac_type":"EI",
              "n_search": 10,
              "boundaries": [[-100,100]],
              "epsilon": 0.01,
              "iteration": 4,
              "minimization":True,
              "optimization":True,
              "n_restart": 5,
              "sampling":np.linspace}

    BayOpt = BayesianOptimization(X, Z, settings, gp, f)
    BayOpt.set_plotter()
    best=BayOpt.run()
    # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
    """best = BayOpt.bayesian_run_min(250,
                                   [[0,1]],
                                   iteration=30,
                                   optimization=False,
                                   plot=False,
                                   n_restart=10,
                                   epsilon=0.01,
                                   sampling=np.linspace)"""

    print("bay:", best)
    #print(BayOpt)


def test_Hartmann_6D():
    dim = 6
    points = 70
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

    gp = GP(x, y, RBF(gradient=False))
    gp.fit()

    settings = {"type": "PSO",
                "ac_type":"EI",
                "n_search": 10,
                "boundaries": [[0, 1] for i in range(6)],
                "epsilon": 0.01,
                "iteration": 30,
                "minimization": True,
                "optimization": True,
                "n_restart": 5,
                "sampling": np.random.uniform}

    BayOpt = BayesianOptimization(x, y, settings, gp, hartmann_6D)

    n_p = 10

    best = BayOpt.run()

    print("Number of points sampled in an iteration: ", n_p ** dim)
    print("bay:", best)


def test_GP_print():
    print(help(GP))
    """dim_test = 2
    dim_out = 1
    n_train_p = 7
    X = np.random.uniform(-2, 2, (40, 2))
    Z = ((X[:, 1] ** 2 * X[:, 0] ** 2) * np.sin((X[:, 1] ** 2 + X[:, 0] ** 2)))[:, None]
    gp = GP(X, Z, RBF())
    gp.get_kernel().plot()
    gp.fit()
    plot = generate_grid(dim_test, 5, [[-3, 3] for i in range(dim_test)])
    gp.optimize_grid(n_points=5, verbose=True)
    gp.get_kernel().plot()
    pred = gp.predict(plot)
    print(gp)
    gp.save_model("/home/merk/Desktop/GP.txt")"""

def plot_test():
    import matplotlib.pyplot as plt
    def f(x):
        x,y=x[:,0],x[:,1]
        return np.sin(x**2+y**2)
    bounds=[[0,3],[0,3]]
    grid=generate_grid(2,50,bounds)
    y=f(grid)
    test_1,test_2=np.meshgrid(np.linspace(0,3,50),np.linspace(0,3,50))
    print(test_1.shape,test_2.shape)
    print(grid[:,0].reshape(50,50))
    cp=plt.contourf(grid[:,0].reshape(50,50),grid[:,1].reshape(50,50),f(grid).reshape(50,50))
    plt.show()

a = time.time()
#test_GP_1D(True)
test_Hartmann_6D()
print("Finished: ", time.time() - a)


