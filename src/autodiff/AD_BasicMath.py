import numpy as np
from src.autodiff.AD_Object import Var

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
    """Returns a new Var with sqrt applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = x.val**(0.5) and der = 0.5*(x.val**(-0.5))*x.der
       Has the form Var(val = x.val**(0.5), der = 0.5*(x.val**(-0.5))*x.der)

    NOTES
    =====
    PRE: 
         - x has AD_Object.Var type
         - Has option to be scalar, vector, or matrix
           - Determined and defined at instantiation
           - More in AD_Object.Var docs
    POST:
         - x is not changed by this function
         - returns a new Var

    EXAMPLES
    =========
    >>> sqrt(Var(1, derivative=1))
    Var(val=1, der=0.5)
    """
    if x.val < 0:
        raise ValueError('Imaginary not implemented, can only sqrt positive numbers')
    try:
        newx = Var(x.val**(0.5), derivative=0.5*(x.val**(-0.5))*x.der)
    except ZeroDivisionError:
        newx = Var(0)
    return newx

def sin(x):
    """Returns a new Var with sine applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = sin(x.val) and der = cos(x.val)*x.der
       Has the form Var(val = sin(x.val), der = cos(x.val)*x.der)

    NOTES
    =====
    PRE: 
         - x has AD_Object.Var type
         - Has option to be scalar, vector, or matrix
           - Determined and defined at instantiation
           - More in AD_Object.Var docs
    POST:
         - x is not changed by this function
         - returns a new Var

    EXAMPLES
    =========
    >>> sin(Var(0, derivative=1))
    Var(val=0, der=1)
    """
    newX = Var(np.sin(x.val), derivative=np.cos(x.val)*x.der)
    return newX

def cos(x):
    """Returns a new Var with cosine applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = cos(x.val) and der = -sin(x.val)*x.der
       Has the form Var(val = cos(x.val), der = -sin(x.val)*x.der)

    NOTES
    =====
    PRE: 
         - x has AD_Object.Var type
         - Has option to be scalar, vector, or matrix
           - Determined and defined at instantiation
           - More in AD_Object.Var docs
    POST:
         - x is not changed by this function
         - returns a new Var

    EXAMPLES
    =========
    >>> cos(Var(0, derivative=1))
    Var(val=1, der=0)
    """
    newX = Var(np.cos(x.val), derivative=-np.sin(x.val)*x.der)
    return newX

def tan(x):
    """Returns a new Var with tangent applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = tan(x.val) and der = sec^2(x.val)*x.der
       Has the form Var(val = cos(x.val), der = -sin(x.val)*x.der) unless
       x.val = c*(pi/2) where c is odd in which case a ValueError exception is raised

    NOTES
    =====
    PRE: 
         - x has AD_Object.Var type
         - Has option to be scalar, vector, or matrix
           - Determined and defined at instantiation
           - More in AD_Object.Var docs
    POST:
         - x is not changed by this function
         - raises a ValueError exception if x.val = c*pi/2 where c is odd
         - returns a new Var

    EXAMPLES
    =========
    >>> tan(Var(0, derivative=1))
    Var(val=0, der=1)
    >>> tan(Var(3*(np.pi)/2, derivative=1))
    ValueError: Tangent undefined at odd multiples of pi/2
    val=3*pi/2
    """
    if (x.val % np.pi > 0) and (x.val % (np.pi/2) == 0):
        raise ValueError(f"Tangent undefined at odd multiples of pi/2\n val={int(x.val/(np.pi/2))}*pi/2")
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