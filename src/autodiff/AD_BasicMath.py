import numpy as np
from src.autodiff.AD_Object import Var

# Returning a new Var for each elementary function
# exponential
# trig functions (sine, cosine, tangent)

def exp(x):
    """Returns a new Var with exp applied to the Var x
    
    :param x: object on which exp is applied, required
    :type x: AD_Object.Var
    :return: new object with exp applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> exp(Var(0))
    Var(val=1, der=1)
    """
    newX = Var(np.exp(x.val), derivative=x.der*(np.exp(x.val)))
    return newX

def log(x, base=2):
    """Returns a new Var with log_base appled to input Var x
    
    :param x: object on which log is applied, required
    :type x: AD_Object.Var
    :param base: base to which the log is used, default=2
    :type base: numtype, required
    :return: new object with log applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> log(Var(2), 2)
    Var(val=1, der=`TODO`)
    """
    return Var(1)

def ln(x):
    """Returns a new Var with ln appled to input Var x
    
    :param x: object on which ln is applied, required
    :type x: AD_Object.Var    
    :return: new object with ln applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> ln(Var(1))
    Var(val=0, der=1)
    """
    return Var(1)
    
def sqrt(x):
    """Returns a new Var with sqrt applied to the input Var x
    
    :param x: object on which sqrt is applied, required
    :type x: AD_Object.Var
    :raises ValueError: when x.val < 0, imaginary numbers have not yet been implemented
    :return: new object with sqrt applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> sqrt(Var(4))
    Var(val=2, der=0.25)
    """
    if x.val < 0:
        raise ValueError('Imaginary not implemented, can only sqrt positive numbers')
    try:
        newX = Var(x.val**(0.5), derivative=0.5*(x.val**(-0.5))*x.der)
    except ZeroDivisionError:
        newX = Var(0)
    return newX

def sin(x):
    """Returns a new Var with sine applied to the input Var x
    
    :param x: object on which sine is applied, required
    :type x: AD_Object.Var
    :return: new object with sine applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> sin(Var(0))
    Var(val=0, der=1)
    """
    newX = Var(np.sin(x.val), derivative=np.cos(x.val)*x.der)
    return newX

def cos(x):
    """Returns a new Var with cosine applied to the input Var x
    
    :param x: object on which cosine is applied, required
    :type x: AD_Object.Var
    :return: new object with cosine applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> cos(Var(0))
    Var(val=1, der=0)
    """
    newX = Var(np.cos(x.val), derivative=-np.sin(x.val)*x.der)
    return newX

def tan(x):
    """Returns a new Var with tangent applied to the input Var x
    
    :param x: object on which tangent is applied, required
    :type x: AD_Object.Var
    :raises ValueError: when x.val is an odd multiple of pi/2, where tangent is undefined
    :return: new object with tangent applied to input
    :rtype: AD_Object.Var

    :example:
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