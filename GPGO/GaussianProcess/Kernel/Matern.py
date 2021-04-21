from .Kernel import Kernel
from numpy import sum, exp
from numpy.linalg import norm
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt


class Matern(Kernel):
    """
    RBF Kernel type class. Type: Kernel, Subtype: RBF
    Init method require the hyperparameters as an input (normal is sigma:2 , l:2)
    """

    def __init__(self, sigma_l=2., l=2., noise=1e-3, gradient=False):
        self.subtype = "matern"
        self.eval_grad= gradient
        super().__init__(hyper={"sigma": sigma_l, "l": l, "noise": noise})

    def kernel_var(self):
        sigma, l, noise = self.gethyper()
        return sigma ** 2 + noise**2

    def product(self, x1, x2=0):
        """
        Kernel product between two parameters
        """
        sigma, l , _ = self.gethyper()
        return sigma ** 2 * (1 +  abs(x1-x2) / l +
                                 1 / 3 *
                             (abs(x1-x2)/l) ** 2 ) * exp(- abs(x1-x2/ l))

    def kernel_product(self, X1, X2):
        """
        Function to compute the vectorized kernel product between X1 and X2
        :param X1: Train data
        :param X2: Distance is computed from those
        :return: np.array
        """

        sigma, l,  noise= self.gethyper()
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2 , axis=1) - 2 * np.dot(X1, X2.T)
        return sigma**2 *(1. + np.sqrt(dist / l**2) + 1/3 * dist / l**2) * np.exp(-  np.sqrt(dist)  / l)


    @staticmethod
    def kernel_(sigma, l, noise, pair_matrix):
        K= sigma**2 *(1. + np.sqrt(pair_matrix / l**2) + 1/3 * pair_matrix / l**2) * np.exp(-  np.sqrt(pair_matrix)  / l)
        K[np.diag_indices_from(K)] += noise ** 2
        return K


    def plot(self):
        """
        Plot the Kernel in 1 Dimension
        """
        X = np.linspace(-4, 4, 100)
        plt.plot(X, self.product(X[:, None]))
        plt.show()

    def sethyper(self, sigma=None, l=None, noise=None):
        """
        Set new hyperparameters value
        :param sigma: Variance
        :param l: Lengthscale
        :param noise: Noise
        """
        if sigma is not None:
            self.hyper["sigma"] = sigma
        if l is not None:
            self.hyper["l"] = l
        if noise is not None:
            self.hyper["noise"] = noise

    '#Get methods '

    def getsubtype(self):
        return self.subtype

    def gethyper_dict(self):
        return self.hyper

    def gethyper(self):
        return tuple(self.hyper.values())

    def get_noise(self):
        return self.hyper["noise"]

    def get_eval(self):
        return self.eval_grad

    def __str__(self):
        kernel_info = f'Kernel type: {self.getsubtype()}\n'
        hyper_info = f'Hyperparameters: {self.gethyper_dict()}\n\n'
        return kernel_info + hyper_info











