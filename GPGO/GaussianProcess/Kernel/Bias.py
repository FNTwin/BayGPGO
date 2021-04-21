from .Kernel import Kernel
from numpy import exp
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt


class Bias(Kernel):
    """
    RBF Kernel object. Type: Kernel, Subtype: RBF
    ...

    Attributes:
    -----------
        hyper      : dict
            Dictionary with sigma_l,l,noise as key
        subtype    : str
            subtype of the kernel , the general type is always kernel while the subtype is RBF
        eval_grad  : bool
            flag to compute the kernel gradient
    Methods
    -----------
    """

    def __init__(self, bias=1., noise=1e-6, gradient=False):
        """
        sigma_l  : float (default 1.)
            float value for the variance of the Kernel
        l        : float (default 1.)
            float value for the lengthscale of the Kernel
        noise    : float (default 1e-6)
            float value for noise of the Gaussian Process
        gradient : bool (default False)
            if set to True it compute the gradient when called to maximize the marginal likelihood"""
        #self.hyper = {"sigma": sigma_l, "l": l, "noise": noise}
        self.subtype = "rbf"
        self.eval_grad = gradient
        super().__init__(hyper={"bias": bias, "noise": noise})

    def kernel_var(self):
        bias, noise = self.gethyper()
        return bias ** 2 + noise ** 2

    @staticmethod
    def kernel_(bias, noise, pair_matrix):
        """
        Static method for computing the Graham Matrix using the kernel, it is require to be defined as it
        is passed in the GP module to optimize the hyperparameters.
        :param sigma: float
            Variance hyperparameter value
        :param l: float
            Variance hyperparameter value
        :param noise: float
            Variance hyperparameter value
        :param pair_matrix: NxM np.array
            Distance matrix between pairs of points
        :return: NxM np.array
            Graham matrix of the Kernel
        """
        K = bias
        K[np.diag_indices_from(K)] += noise ** 2
        return K

    @staticmethod
    def kernel_eval_grad_(bias, noise, pair_matrix):
        """
            Static method for computing the derivates of the Kernel, it is require to be defined if __eval_grad is set
            to True as it is passed in the GP module to optimize the hyperparameters with the use of the gradient.
            :param sigma: float
                Variance hyperparameter value
            :param l: float
                Variance hyperparameter value
            :param noise: float
                Variance hyperparameter value
            :param pair_matrix: NxM np.array
                Distance matrix between pairs of points
            :return: NxM np.array
                Graham matrix of the Kernel
        """
        K = bias
        K_norm = K
        K_norm[np.diag_indices_from(K)] += noise ** 2
        K_1 = 1
        K_3 = np.eye(pair_matrix.shape[0]) * noise * 2
        return (K_norm, K_1,  K_3)

    def product(self, x1, x2=0):
        """
        Methods used to compute the kernel values on single points.
        x1 : np.array
        x2 : np.array (default 0)
        :return: kernel value between two data point x1,x2
        """
        bias, noise = self.gethyper()
        return bias

    def kernel_product(self, X1, X2):
        """
        Vectorized method to compute the kernel product between X1 and X2 by creating a distance matrix of the X1 and X2
        entry and then computing the Kernel. The Kernel is computing respect the X2 points.
        X1: np.array of shape N,.
            N points data
        X2: np.array of shape M,.
           M points data
        return : np.array of dimension NxM
            return the Graham Matrix of the Kernel
        """
        bias, noise = self.gethyper()
        dist = np.sum(X1 ** 2, axis=1)[:, None] + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return bias

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

    '#Get methods'
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




