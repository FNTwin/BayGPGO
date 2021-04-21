from .Operations import Add, Mul


class Kernel:
    """
    Parent class for kernels. It's used to check if the type of every custom kernel is right.
    """

    def __init__(self, *args, **kwargs):
        self.__type = "Kernel"
        self._tree=None
        self._hyper_space = Hyperparameters(**kwargs["hyper"])

    def gettype(self):
        return self.__type

    def __add__(self, other):
        return Add(self,other)

    def __mul__(self, other):
        return Mul(self,other)

    @property
    def hyper(self):
        return self._hyper_space.hyper


class Hyperparameters:
    def __init__(self, **kwargs):
        self.store=kwargs
        self.create_hyper_dict(**self.store)

    def create_hyper_dict(self, **kwargs):
        self.hyper={}
        for name in kwargs:
            self.hyper[name]=kwargs[name]



