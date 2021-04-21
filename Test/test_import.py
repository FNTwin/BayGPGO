import numpy as np
from GPGO import GP
from GPGO import RBF
from GPGO import BayesianOptimization
from GPGO import dpd_opt_script

dpd_opt_script.main()

"""x=np.random.uniform(0,10,3)[:,None]
y=x*3
a=GP(x,y,RBF())
print(a)
"""