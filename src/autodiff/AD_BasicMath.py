
from src.autodiff.AD_Object import Var
import numpy as np

# Returning a new Var for each elementary function
# exponential
# trig functions (sine, cosine, tangent)

def exp(x):
    raise NotImplementedError

def log(x):
    raise NotImplementedError

def ln(x):
    raise NotImplementedError

def sqrt(x):
    raise NotImplementedError

def sin(x):
    newX = Var(np.sin(x.val), derivative=np.cos(x.val)*x.der)
    # check our notation for instantiating a variable -- just AD_Object(value, derivative_seed_value)?
    return newX

def cos(x):
    newX = Var(np.cos(x.val), derivative=-np.sin(x.val)*x.der)
    return newX

def tan(x):
    newX = Var(np.tan(x.val), derivative=(1/np.cos(x.val)**2)*x.der)
    return newX

def csc(x):
    raise NotImplementedError

def sec(x):
    raise NotImplementedError

def cot(x):
    raise NotImplementedError

def arcsin(x):
    raise NotImplementedError

def arccos(x):
    raise NotImplementedError

def arctan(x):
    raise NotImplementedError