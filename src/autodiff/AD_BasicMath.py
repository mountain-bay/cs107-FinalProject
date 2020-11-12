
from AD_Object import Var
import numpy as np

# Returning a new Var for each elementary function
# exponential
# trig functions (sine, cosine, tangent)

def exp(x):
    pass

def log(x):
    pass

def ln(x):
    pass

def sqrt(x):
    pass

def sin(x):
    newX = Var(np.sin(x.val), np.cos(x.val)*x.der)
    # check our notation for instantiating a variable -- just AD_Object(value, derivative_seed_value)?
    return newX

def cos(x):
    newX = Var(np.cos(x.val), -np.sin(x.val)*x.der)
    return newX

def tan(x):
    newX = Var(np.tan(x.val), (1/np.cos(x.val)**2)*x.der)
    return newX

def csc(x):
    pass

def sec(x):
    pass

def cot(x):
    pass

def arcsin(x):
    pass

def arccos(x):
    pass

def arctan(x):
    pass