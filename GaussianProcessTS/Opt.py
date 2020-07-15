import numpy as np
from GaussianProcess.GP import GP, generate_grid
from GaussianProcess.Plotting import plot_BayOpt
from GaussianProcess.util import time_log, log_bo, Observer
from scipy.optimize import minimize, fmin_l_bfgs_b
from scipy.stats import norm
from DIRECT import solve
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import time

#settings={type_of_opt, n_search, boundaries, iteration, minimization,  epsilon, optimization_GP, n_restart, func_sampling}

class BayesianOptimization():

    def __init__(self, X: np.ndarray, Y: np.ndarray, settings=None, GP=None, func=None, err=1e-6):
        self.__dim_input = X[0].shape[0]
        self.__dim_output = Y[0].shape[0]
        self.__dimension = X.shape[0]
        self.__X = X
        self.__Y = Y
        self.__settings=settings
        self.__GP = GP
        self.__func = func
        self.__err = err
        self.__it = None
        self.__time = time_log()
        self.__observer=Observer("Bayesian Optimization")
        self.__old_data = [X, Y]

    def run(self):
        print(self.__settings)
        try:
            if self.__settings is not None:
                 copy_settings=self.__settings
                 bay_opt_methods= {"DIRECT":self.bayesian_run_DIRECT,
                                     "BFGL":self.bayesian_run_BFGL,
                                    "NAIVE": self.bayesian_run_min}

                 optimizer=bay_opt_methods[self.get_info("type")]
                 self.__observer.type=self.get_info("type")
                 del copy_settings["type"]
                 print(copy_settings)
                 return optimizer(**copy_settings)
            else:
                print("Settings not specified")
        except:
            print("Error on inizialing the Bayesian Optimization")

    def get_info(self,key):
        return self.__settings[key]

    def next_sample_validation(self, new_sample, boundaries):
        #if np.any(np.all(np.abs(new_sample - self.get_X()) < self.get_err(), axis=1)):
        #if np.sqrt(np.sum((self.get_X()[-1, :] - new_sample)**2))< self.get_err():
        if np.any(np.sqrt(np.sum((self.get_X() - new_sample) ** 2,axis=1)) < self.get_err()):
            print("+++++++++++++++++++++++++++++++++++++++++++")
            print(new_sample)
            print(new_sample.shape[0])
            return np.random.uniform(boundaries[:, 0], boundaries[:, 1], (1, self.get_dim_inputspace()))

        else:
            return new_sample

    def compute_new_sample(self, new_sample):
        func = self.get_func()

        try:
            sample_Y = func(new_sample)

        except:
            sample_Y = func(np.expand_dims(new_sample, axis=0))

        return sample_Y

    '#===================================BAYESIAN DIRECT====================================='

    def optimize(self, f, boundaries, max_iter=10000):

        def DIRECT_wrapper(f):
            def g(x, user_data):
                return -f(np.array([x])),0
            return g

        lb=boundaries[:, 0]
        ub=boundaries[:,1]
        x, val, _ = solve(DIRECT_wrapper(f), lb, ub, maxf= 80000 , maxT=max_iter, algmethod=1)
        print("DIRECT:" ,x, val)
        return x

    def Expected_improment_dir(self, point):
        """def Z(point, mean, variance):
            return (-mean + point + epsilon) / variance
        self.get_GP().fit()

        mean, variance = self.get_GP().predict(point)
        new_point=np.min(self.get_Y())
        epsilon= 0.01

        EI = (-mean + new_point + epsilon) * norm.cdf(Z(new_point, mean, variance)) \
             + variance * norm.pdf(Z(new_point, mean, variance))

        EI[variance == 0] = 0
        return -EI"""
        def Z(point, mean, variance, epsilon):
            return (-mean + point + epsilon) / variance

        epsilon=0.01

        mean, variance = self.get_GP().predict(point)
        point = np.min(self.get_Y())

        EI = (-mean + point * epsilon) * norm.cdf(Z(point, mean, variance, epsilon)) \
             + variance * norm.pdf(Z(point, mean, variance, epsilon))
        EI[EI == 0] = 0
        return EI

    def DIRECT_new_sample_loc(self, boundaries, search_grid, epsilon):

        min = self.optimize(self.Expected_improment_dir, boundaries)
        return np.atleast_2d(min)

    def bayesian_run_DIRECT(self,
                            n_search,
                            boundaries,
                            iteration=20,
                            epsilon=0.01,
                            minimization=True,
                            optimization=False,
                            n_restart=100,
                            sampling=np.random.uniform):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")
        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            boundaries_array = np.asarray(boundaries)

            for i in range(iteration):
                print("iteration: ", i+1)

                search_grid = generate_grid(dim, n_search, boundaries, function=sampling)

                if optimization:
                    gp.opt(n_restarts=n_restart)
                    print("Optimization: ", i, " completed")

                gp.fit()

                predicted_best_X = self.DIRECT_new_sample_loc(boundaries_array, search_grid, epsilon)

                # Check if it is a duplicate
                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)
                print("COMPUTING:", predicted_best_X)
                predicted_best_Y = self.compute_new_sample(predicted_best_X)
                self.observe(predicted_best_Y)

                print(i, "It: ", predicted_best_X, " , Y: ", predicted_best_Y)

                # Augment the dataset of the BO and the GP objects
                self.augment_XY(predicted_best_X, predicted_best_Y)
                gp.augment_XY(predicted_best_X, predicted_best_Y)

            if minimization:
                best_index = np.argmin(self.get_Y())
            else:
                best_index = np.argmax(self.get_Y())

            self.observer_plot()
            return self.get_X()[best_index], self.get_Y()[best_index]




    '#===================================Bayesian BFGL======================================='

    def bayesian_run_BFGL(self,
                          n_search,
                          boundaries,
                          iteration=10,
                          epsilon=0.01,
                          minimization=True,
                          optimization=False,
                          n_restart=100,
                          sampling=np.random.uniform):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")
        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            boundaries_array = np.asarray(boundaries)

            for i in range(1, iteration + 1):
                print("Iteration: ", i)
                # Generate dimensional Grid to search
                search_grid = generate_grid(dim, n_search, boundaries, function=sampling)

                # Generate surrogate model GP and predict the grid values
                gp.fit()
                if optimization:
                    gp.opt(n_restarts=n_restart)
                    print("Optimization: ", i, " completed")

                # Compute the EI and the new theoretical best
                """predicted_best_X = self.propose_new_sample_loc(self.Expected_improment, gp,
                                                               boundaries_array, search_grid, epsilon)"""
                predicted_best_X = self.propose_new_sample_loc(self.Expected_improment, gp,
                                                               boundaries_array, n_search, epsilon)

                # Check if it is a duplicate
                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)
                predicted_best_Y = self.compute_new_sample(predicted_best_X)

                self.observe(predicted_best_Y)

                print(i, "It: ", predicted_best_X, " , Y: ", predicted_best_Y)

                # Augment the dataset of the BO and the GP objects
                self.augment_XY(predicted_best_X, predicted_best_Y)
                gp.augment_XY(predicted_best_X, predicted_best_Y)

            if minimization:
                best_index = np.argmin(self.get_Y())
            else:
                best_index = np.argmax(self.get_Y())

            self.observer_plot()

            return self.get_X()[best_index], self.get_Y()[best_index]

    def Expected_improment(self, new_points, max, gp, epsilon):
        print(new_points)
        def Z(point, mean, variance, epsilon):
            return (-mean + point + epsilon) / variance

        mean, variance = gp.predict(new_points)
        point = max

        EI = (-mean + point + epsilon) * norm.cdf(Z(point, mean, variance, epsilon)) \
             + variance * norm.pdf(Z(point, mean, variance, epsilon))
        EI[EI == 0] = 0
        return EI

    def propose_new_sample_loc(self, EI_func, gp, boundaries, n_search_points, epsilon):
        dim = self.get_dim_inputspace()
        min_val = None
        max = np.min(self.get_Y())

        """def min_objective(X, max, gp, epsilon):
            return -EI_func(X.reshape(-1, dim), max, gp, epsilon)"""

        #improvement = EI_func(n_search_points, max, gp, epsilon)
        min_x = None

        for i in np.random.uniform(boundaries[:, 0], boundaries[:, 1], size=(n_search_points, dim)):


            res = minimize(lambda X: -EI_func(X.reshape(-1, dim), max=max, gp=gp, epsilon=epsilon),
                           x0=i, bounds=boundaries, method='L-BFGS-B' )


            if not res.success:
                continue

            if min_val is None or res.fun[0] < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x

    '#===================================Bayesian MINIMIZE======================================='


    def bayesian_run_min(self, n_search,
                         boundaries,
                         iteration=10,
                         epsilon=0.01,
                         minimization=True,
                         optimization=False,
                         n_restart=5,
                         sampling=np.random.uniform,
                         plot=False):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:

            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            self.__it = iteration
            boundaries_array = np.asarray(boundaries)

            if minimization:

                for i in range(1, iteration + 1):
                    print("Iteration: ", i)
                    # Generate dimensional Grid to search
                    if sampling== "LHS":
                        sampling = LHS(xlimits=boundaries_array)
                        search_grid=sampling(n_search)

                    else:
                        search_grid = generate_grid(dim, n_search, boundaries, sampling)

                    # Generate surrogate model GP and predict the grid values
                    gp.fit()
                    if optimization:
                        #gp.optimize(constrains=opt_constrain, n_points=n_opt_points, function=np.random.uniform)
                        gp.opt(n_restarts=n_restart)
                        print("Optimization: ", i, " completed")

                    mean, var = gp.predict(search_grid)

                    print("Surrogate Model generated: ", i)

                    # Compute the EI and the new theoretical best
                    predicted_best_X, improvements, best_value = self.optimization_min(search_grid,
                                                                                       mean,
                                                                                       var,
                                                                                       epsilon,
                                                                                       plot)
                    tm.time()

                    # Check if it is a duplicate
                    predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)

                    if self.get_func() is not None:
                        predicted_best_Y = self.compute_new_sample(predicted_best_X)
                        self.observe(predicted_best_Y)
                        # Augment the dataset of the BO and the GP objects
                        self.augment_XY(predicted_best_X, predicted_best_Y)
                        gp.augment_XY(predicted_best_X, predicted_best_Y)

                    else:
                        raise ValueError('Function not defined. If you are running an optimization' +
                                         ' with an external function use the command bayesian_run_min_single')

                best_index = np.argmin(self.get_Y())
                tm.time_end()
                # log_bo(self.__str__())
                print("TIME:",tm)
                self.observer_plot()

                return self.get_X()[best_index], self.get_Y()[best_index]

            else:
                return self.bayesian_run_max(n_search,
                                        boundaries,
                                        iteration,
                                        epsilon,
                                        minimization,
                                        optimization,
                                        n_restart,
                                        sampling,
                                        plot=False)


    def optimization_min(self, search_grid, mean, variance, epsilon, plot):
        best = np.min(self.get_Y())
        improvement = Expected_improment_min(best, mean, variance, epsilon)
        new_prediction = search_grid[np.argmax(improvement)]

        if plot:
            args = [self.get_X(), self.get_Y(), search_grid, mean, variance, improvement]
            plt = plot_BayOpt(*args)
            plt.axvline(new_prediction, label="Suggested Point", linestyle="--",
                        color="red", alpha=0.4)
            plt.legend()
            plt.show()

        return new_prediction, improvement, best

    '#===================================Bayesian MAXIMIZE======================================='

    def bayesian_run_max(self, n_search,
                         boundaries,
                         iteration=10,
                         epsilon=0.1,
                         minimization=False,
                         optimization=False,
                         n_restart=5,
                         sampling=np.random.uniform,
                         plot=False):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            self.__it = iteration
            boundaries_array = np.asarray(boundaries)

            for i in range(1, iteration + 1):
                print("Iteration: ", i)
                # Generate dimensional Grid to search
                if sampling == "LHS":
                    sampling = LHS(xlimits=boundaries_array)
                    search_grid = sampling(n_search)

                else:
                    search_grid = generate_grid(dim, n_search, boundaries, sampling)

                # Generate surrogate model GP and predict the grid values
                gp.fit()
                if optimization:
                    gp.opt(n_restarts=n_restart)
                    print("Optimization: ", i, " completed")
                mean, var = gp.predict(search_grid)
                print("Surrogate Model generated: ", i)

                # Compute the EI and the new theoretical best
                predicted_best_X, improvements, best_value = self.optimization_max(search_grid,
                                                                                   mean,
                                                                                   var,
                                                                                   epsilon,
                                                                                   plot)
                tm.time()

                # Check if it is a duplicate
                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)
                predicted_best_Y = self.compute_new_sample(predicted_best_X)
                self.__observer.observe(predicted_best_Y)

                # Augment the dataset of the BO and the GP objects
                self.augment_XY(predicted_best_X, predicted_best_Y)
                gp.augment_XY(predicted_best_X, predicted_best_Y)

            best_index = np.argmax(self.get_Y())
            tm.time_end()
            # log_bo(self.__str__())
            self.__observer.plot()

            return self.get_X()[best_index], self.get_Y()[best_index]

    def optimization_max(self, search_grid, mean, variance, epsilon, plot):

        best = np.max(self.get_Y())
        improvement = Expected_Improvement_max(best, mean, variance, epsilon)
        new_prediction = search_grid[np.argmax(improvement)]

        if plot:
            args = [self.get_X(), self.get_Y(), search_grid, mean, variance, improvement]
            plt = plot_BayOpt(*args)
            plt.axvline(new_prediction, label="Suggested Point", linestyle="--",
                        color="red", alpha=0.4)
            plt.legend()
            plt.show()
        return new_prediction, improvement, best

    '#=====================SINGLE MIN/MAX WITH EXTERNAL FUNCTION========================'

    def bayesian_run_single(self, n_search_points,
                            boundaries,

                            optimization=False,
                            minimization=True,
                            epsilon=0.01,
                            opt_constrain=[[1, 30], [1, 30], [1e-4,2]],
                            n_opt_points=100,
                            func=np.random.uniform):

        """Devo fare una routine per formarte/leggere i file per il gp e la bo
        in questo caso carica i dati del gp gia aggiornati da un foglio, sara da mettere una pipeline"""
        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            self.__it = None
            boundaries_array = np.asarray(boundaries)

            print("Generating surrogate model\n")
            # Generate dimensional Grid to search
            if func == "LHS":
                sampling = LHS(xlimits=boundaries_array)
                search_grid = sampling(n_search_points)

            else:
                #search_grid = generate_grid(dim, n_search_points, boundaries, func)
                search_grid=np.random.uniform(-10,140,(n_search_points,dim))

            # Generate surrogate model GP and predict the grid values
            gp.fit()
            if optimization:
                gp.optimize(constrains=opt_constrain,
                            n_points=n_opt_points,
                            function=np.random.uniform)
                print("Optimization of the model completed\n")

            mean, var = gp.predict(search_grid)
            print("Surrogate Model generated\n")

            # Compute the EI and the new theoretical best
            if minimization:
                predicted_best_X, improvements, best_value = self.optimization_min(search_grid,
                                                                                   mean,
                                                                                   var,
                                                                                   epsilon,
                                                                                   plot=False)

            else:
                predicted_best_X, improvements, best_value = self.optimization_max(search_grid,
                                                                                   mean,
                                                                                   var,
                                                                                   epsilon,
                                                                                   plot=False)
            # Check if it is a duplicate
            predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)

            return predicted_best_X

    def bayesian_run_single_DIRECT(self, n_search_points,
                            boundaries,
                            optimization=False,
                            epsilon=0.01,
                            opt_constrain=[[1, 30], [1, 30], [1e-4,2]],
                            n_opt_points=100,
                            func=np.random.uniform):
        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            self.__it = None
            boundaries_array = np.asarray(boundaries)

            print("Generating surrogate model\n")
            # Generate dimensional Grid to search
            if func == "LHS":
                sampling = LHS(xlimits=boundaries_array)
                search_grid = sampling(n_search_points)

            else:
                search_grid = generate_grid(dim, n_search_points, boundaries, func)


            # Generate surrogate model GP and predict the grid values
            if optimization:
                gp.optimize(constrains=opt_constrain,
                            n_points=n_opt_points,
                            function=np.random.uniform)
                print("Optimization of the model completed\n")

            gp.fit()

            #mean, var = gp.predict(search_grid)
            print("Surrogate Model generated\n")

            predicted_best_X = self.DIRECT_new_sample_loc(boundaries_array, search_grid, epsilon)

            # Check if it is a duplicate
            predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)

            return predicted_best_X


    '#=====================================UTILITIES============================================'

    def augment_dataset(self, dataset, new_data):
        try:
            dataset.shape[1] == new_data.shape[1]
            return np.concatenate((dataset, new_data))

        except:

            try:
                dataset.shape[1] == new_data.shape[0]
                return np.concatenate((dataset, np.expand_dims(new_data, axis=0)))

            except:
                raise ValueError(f'Data dimensions are wrong: {dataset.shape} != {new_data.shape}')

    def augment_X(self, new_data):
        self.__X = self.augment_dataset(self.get_X(), new_data)

    def augment_Y(self, new_data):
        self.__Y = self.augment_dataset(self.get_Y(), new_data)

    def augment_XY(self, new_data_X, new_data_Y):
        self.augment_X(new_data_X)
        self.augment_Y(new_data_Y)

    def set_func(self, func):
        self.__func = func

    def get_X(self):
        return self.__X

    def get_Y(self):
        return self.__Y

    def get_GP(self):
        return self.__GP

    def get_func(self):
        return self.__func

    def get_dim_data(self):
        return self.__dimension

    def get_dim_inputspace(self):
        return self.__dim_input

    def get_dim_outspace(self):
        return self.__dim_output

    def get_time_logger(self):
        return self.__time

    def get_err(self):
        return self.__err

    def set_err(self, err):
        self.__err = err

    def __str__(self):
        header = "============================================================================\n"
        old_data = f'Bayesian Run initialized with: {self.__it} iterations\nDATASET\n{self.__old_data}\n'
        old_data += "----------------------------------------------------------------------------\n\n"
        time = f'Time: {str(self.get_time_logger().total())}\n\n'
        count = 0
        datanew = f'Iteration data\n\n'
        X, Y = self.get_X(), self.get_Y()
        for i in range(X.shape[0]):
            count += 1
            datanew += f'{count}\t{X[i]}\t{Y[i]}\n'

        return header + old_data + time + datanew + header

    '#=================================TEST ROUTINES============================================'

    @staticmethod
    def test_long(x,y,f, n_search_points,
                 boundaries,
                 iter,
                 minima,
                 opt=False):

        gp_EI = GP(x, y, noise=2e-5)
        gp_EI.fit()
        gp_DIRECT=GP(x, y,noise=2e-5)
        gp_DIRECT.fit()
        BayOpt_EI = BayesianOptimization(x, y, gp_EI, f, err=1e-3)
        BayOpt_DIRECT= BayesianOptimization(x, y, gp_DIRECT, f, err=1e-3)

        # best=BayOpt.bayesian_run(100,  [[-1,4] for i in range(dim_test)] , iteration=30, optimization=False)
        """err, val = BayOpt_EI.test_min(n_search_points,
                 boundaries,
                 minima,
                 iteration=iter,
                 optimization=opt,
                 epsilon=0.01,
                 opt_constrain=[[0.5, 30], [0.5, 30]],
                 n_opt_points=100,
                 func=np.random.uniform,
                 plot=False)

        print(val[-1])"""

        err_DIRECT, val_DIRECT=BayOpt_DIRECT.direct_test(n_search_points,
                 boundaries,
                 minima,
                 iteration=iter,
                 optimization=opt,
                 epsilon=0.01,
                 opt_constrain=[[2, 30], [2, 30]],
                 n_opt_points=100,
                 func=np.random.uniform,
                 plot=False)

        plt.scatter(np.arange(0, iter, 1), val_DIRECT, marker="o", color="orange", label="DIRECT")
        plt.plot(np.arange(0, iter, 1), np.full(iter, minima), color="black", linestyle="--", label="Minima")
        #plt.scatter(np.arange(0, iter, 1), val, marker="o", color="red",  label="EI")
        plt.xlabel("Iteration")
        plt.ylabel("Function value")
        plt.legend()
        plt.show()

    def direct_test(self,n_search_points,
                 boundaries,
                 minima,
                 iteration=10,
                 optimization=False,
                 epsilon=0.1,
                 opt_constrain=[[2, 30], [2, 30]],
                 n_opt_points=100,
                 func=np.random.uniform,
                 plot=False):
        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            self.__it = iteration
            boundaries_array = np.asarray(boundaries)

            error = []
            val = []

            for i in range(1, iteration + 1):
                print("Iteration: ", i)

                search_grid = generate_grid(dim, n_search_points, boundaries, function=func)

                if optimization:
                    gp.optimize(constrains=opt_constrain, n_points=n_opt_points, function=func)
                    print("Optimization: ", i, " completed")
                gp.fit()

                predicted_best_X = self.DIRECT_new_sample_loc(boundaries_array, search_grid, epsilon)

                # Check if it is a duplicate
                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)
                print("COMPUTING:", predicted_best_X)
                predicted_best_Y = self.compute_new_sample(predicted_best_X)

                print(i, "It: ", predicted_best_X, " , Y: ", predicted_best_Y)

                # Augment the dataset of the BO and the GP objects
                self.augment_XY(predicted_best_X, predicted_best_Y)
                gp.augment_XY(predicted_best_X, predicted_best_Y)
                error.append(np.abs(np.min(self.get_Y()) - minima))
                val.append(np.min(self.get_Y()))

            """plt.scatter(np.arange(0, iteration, 1), val, marker="o", color="orange", label="DIRECT")
            plt.plot(np.arange(0, iteration, 1), np.full(iteration, minima), color="black", label="Minima")
            plt.xlabel("Iteration")
            plt.ylabel("Function value")
            plt.legend()
            plt.show()"""

            return error, val



    def test_min(self, n_search_points,
                 boundaries,
                 minima,
                 iteration=10,
                 optimization=False,
                 epsilon=0.1,
                 opt_constrain=[[2, 30], [2, 30]],
                 n_opt_points=100,
                 func=np.random.uniform,
                 plot=False):

        if GP is None:
            raise ValueError("Gaussian Process not existing. Define one before running a " +
                             "Bayesian Optimization")

        else:
            gp = self.get_GP()
            dim = self.get_dim_inputspace()
            tm = self.get_time_logger()
            self.__it = iteration
            boundaries_array = np.asarray(boundaries)

            error = []
            val = []

            for i in range(1, iteration + 1):
                print("Iteration: ", i)
                # Generate dimensional Grid to search

                search_grid = generate_grid(dim, n_search_points, boundaries, func)

                # Generate surrogate model GP and predict the grid values
                gp.fit()
                if optimization:
                    if (i % 5 == 0):
                        gp.optimize(constrains=opt_constrain, n_points=n_opt_points, function=np.random.uniform)
                        print("Optimization: ", i, " completed")
                    else:
                        pass
                mean, var = gp.predict(search_grid)

                print("Surrogate Model generated: ", i)

                # Compute the EI and the new theoretical best
                predicted_best_X, improvements, best_value = self.optimization_min(search_grid,
                                                                                   mean,
                                                                                   var,
                                                                                   epsilon,
                                                                                   plot)
                tm.time()

                # Check if it is a duplicate

                predicted_best_X = self.next_sample_validation(predicted_best_X, boundaries_array)

                if self.get_func() is not None:
                    predicted_best_Y = self.compute_new_sample(predicted_best_X)
                    # Augment the dataset of the BO and the GP objects
                    self.augment_XY(predicted_best_X, predicted_best_Y)
                    gp.augment_XY(predicted_best_X, predicted_best_Y)
                    error.append(np.abs(np.min(self.get_Y()) - minima))
                    val.append(np.min(self.get_Y()))

                else:
                    raise ValueError('Function not defined. If you are running an optimization' +
                                     ' with an external function use the command bayesian_run_min_single')

            best_index = np.argmin(self.get_Y())
            tm.time_end()
            # log_bo(self.__str__())

            """plt.scatter(np.arange(0, iteration, 1), val, marker="o", color="red", label="EI")
            plt.plot(np.arange(0, iteration, 1), np.full(iteration, minima), color="black", label="Minima")
            plt.xlabel("Iteration")
            plt.ylabel("Function value")
            plt.legend()
            plt.show()"""

            return error, val

    def observe(self,value):
        self.__observer.observe(value)

    def observer_plot(self):
        self.__observer.plot()


def Expected_Improvement_max(new_point, mean, variance, epsilon):
    def Z(point, mean, variance):
        return (mean - point - epsilon) / variance

    EI = (mean - new_point - epsilon) * norm.cdf(Z(new_point, mean, variance)) \
         + variance * norm.pdf(Z(new_point, mean, variance))
    EI[variance == 0] = 0
    return EI


def Expected_improment_min(new_point, mean, variance, epsilon):
    def Z(point, mean, variance):
        return (-mean + point + epsilon) / variance

    EI = (-mean + new_point + epsilon) * norm.cdf(Z(new_point, mean, variance)) \
         + variance * norm.pdf(Z(new_point, mean, variance))

    EI[variance == 0] = 0
    return EI

