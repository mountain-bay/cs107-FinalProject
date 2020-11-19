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


def log(x):
    """Returns a new Var with log base 10 applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = log(x.val) and der = 1/(x.val*ln(10))*x.der
       Has the form Var(val = log(x.val), der = 1/(x.val*np.log(10))*x.der)

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
    >>> log(Var(10, derivative=np.log(10)))
    Var(val=1, der=1)
    """
    if (x.val < 0):
        raise ValueError("Log undefined at negative values")
    newX = Var(np.log10(x.val), derivative=1/(x.val*np.log(10))*x.der)
    return newX


def ln(x):
    """Returns a new Var with ln applied to the Var x

    INPUTS
    =======
    x: AD_Object.Var, required

    RETURNS
    ========
    newX: AD_Object.Var with val = ln(x.val) and der = 1/(x.val)*x.der
       Has the form Var(val = ln(x.val), der = 1/(x.val)*x.der)

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
    >>> ln(Var(np.e, derivative=np.e))
    Var(val=1, der=1)
    """
    if (x.val < 0):
        raise ValueError("Ln undefined at negative values")
    newX = Var(np.log(x.val), derivative=1/(x.val)*x.der)
    return newX


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
        raise ValueError(
            'Imaginary not implemented, can only sqrt positive numbers')
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
        raise ValueError(
            f"Tangent undefined at odd multiples of pi/2\n val={int(x.val/(np.pi/2))}*pi/2")
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
