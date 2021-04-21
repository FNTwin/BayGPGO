from numpy import sum, exp
from numpy.linalg import norm
from numpy import ndarray
import numpy as np


class Add():
    def __init__(self, Kernel1, Kernel2, gradient=False, diff=False):
        super().__init__()
        self._lkernel=Kernel1
        self._rkernel=Kernel2
        self.eval_grad = gradient
        self.diff_hypers=diff

    def __str__(self):
        return str(self._lkernel)+str(self._rkernel)

    def __add__(self, other):
        return Add(self,other)

    def __mul__(self, other):
        return Mul(self,other)


class Mul():
    def __init__(self, Kernel1, Kernel2, gradient=False, diff=False):
        super().__init__()
        self._lkernel=Kernel1
        self._rkernel=Kernel2
        self.eval_grad = gradient
        self.diff_hypers=diff

    def __str__(self):
        return str(self._lkernel)+str(self._rkernel)

    def __mul__(self, other):
        return Mul(self,other)

    def __add__(self, other):
        return Add(self,other)

