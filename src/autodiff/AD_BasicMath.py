import numpy as np
from .AD_Object import Var

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
    >>> from autodiff.AD_BasicMath import exp
    >>> from autodiff.AD_Object import Var
    >>> exp(Var(0))
    Var(val=1.0, der=1.0)
    """
    newX = Var(np.exp(x.val), derivative=x.der*(np.exp(x.val)))
    return newX
  
def log(x, base=10):
    """Returns a new Var with log_base appled to input Var x, default base is 10
    
    :param x: object on which log is applied, required
    :type x: AD_Object.Var
    :param base: base to which the log is used, default=2
    :type base: numtype, required
    :return: new object with log applied to input
    :rtype: AD_Object.Var
    
    :examples:
    >>> import numpy as np
    >>> from autodiff.AD_Object import Var
    >>> from autodiff.AD_BasicMath import log
    >>> log(Var(1, derivative=np.log(10)))
    Var(val=0.0, der=0.9999999999999999)
    >>> log(Var(1), 2)
    Var(val=0.0, der=1/np.log(2))
    """
    if (x.val < 0):
        raise ValueError("Log undefined at negative values")    
    newX = Var(np.log(x.val)/np.log(base), derivative=1/(x.val*np.log(base))*x.der)
    return newX

def ln(x):
    """Returns a new Var with ln appled to input Var x
    
    :param x: object on which ln is applied, required
    :type x: AD_Object.Var    
    :return: new object with ln applied to input
    :rtype: AD_Object.Var
    
    :examples:
    >>> import numpy as np
    >>> from autodiff.AD_Object import Var
    >>> from autodiff.AD_BasicMath import ln
    >>> ln(Var(np.e, derivative=np.e))
    Var(val=1.0, der=1.0)
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
    >>> from autodiff.AD_BasicMath import sqrt
    >>> from autodiff.AD_Object import Var
    >>> sqrt(Var(4))
    Var(val=2.0, der=0.25)
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
    >>> from autodiff.AD_BasicMath import sin
    >>> from autodiff.AD_Object import Var
    >>> sin(Var(0))
    Var(val=0.0, der=1.0)
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
    >>> from autodiff.AD_BasicMath import cos
    >>> from autodiff.AD_Object import Var
    >>> cos(Var(0))
    Var(val=1.0, der=-0.0)
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
    >>> from autodiff.AD_BasicMath import tan
    >>> from autodiff.AD_Object import Var
    >>> tan(Var(0, derivative=1))
    Var(val=0.0, der=1.0)
    """
    if (x.val % np.pi > 0) and (x.val % (np.pi/2) == 0):
        raise ValueError(f"Tangent undefined at odd multiples of pi/2\n val={int(x.val/(np.pi/2))}*pi/2")
    newX = Var(np.tan(x.val), derivative=(1/np.cos(x.val)**2)*x.der)
    return newX

def csc(x):
    """Returns a new Var with cosecant applied to the input Var x
    
    :param x: object on which cosecant is applied, required
    :type x: AD_Object.Var
    :return: new object with cosecant applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> csc(Var(np.pi/2))
    Var(val=1.0, der=6.123234e-17)
    """
    if (x.val == 0) or (x.val % (np.pi) == 0):
        raise ValueError("Cosecant undefined at 0 and multiples of pi")
    newX = Var(1/np.sin(x.val), derivative=-1/np.tan(x.val)*(1/np.sin(x.val))*x.der)
    return newX

def sec(x):
    """Returns a new Var with secant applied to the input Var x
    
    :param x: object on which secant is applied, required
    :type x: AD_Object.Var
    :return: new object with secant applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> sec(Var(2*np.pi/3))
    Var(val=-2.0, der=3.46410161514)
    """
    if (x.val % np.pi > 0) and (x.val % (np.pi/2) == 0):
        raise ValueError("Secant undefined at odd multiples of pi/2")
    newX = Var(1/np.cos(x.val), derivative = np.tan(x.val)/np.cos(x.val)*x.der)
    return newX

def cot(x):
    """Returns a new Var with cotangent applied to the input Var x
    
    :param x: object on which cotangent is applied, required
    :type x: AD_Object.Var
    :return: new object with cotangent applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> cot(Var(np.pi/4))
    Var(val=1.0, der=-2.0)
    """
    if (x.val == 0) or (x.val % (np.pi) == 0):
        raise ValueError("Cotangent undefined at 0 and multiples of pi")
    newX = Var(1/np.tan(x.val), derivative = -1/(np.sin(x.val)**2)*x.der)
    return newX

def arcsin(x):
    """Returns a new Var with arcsine applied to the input Var x
    
    :param x: object on which arcsine is applied, required
    :type x: AD_Object.Var
    :return: new object with arcsine applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> arcsine(Var(0))
    Var(val=0, der=1.0)
    """
    if (x.val >= 1) or (x.val < -1):
        raise ValueError("Arcsine undefined at x=1, values greater than 1 or less than -1")
    newX = Var(np.arcsin(x.val), derivative = 1/np.sqrt(1-x.val**2)*x.der)
    return newX

def arccos(x):
    """Returns a new Var with arccosine applied to the input Var x
    
    :param x: object on which arccosine is applied, required
    :type x: AD_Object.Var
    :return: new object with arccosine applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> arccos(Var(np.pi/2))
    Var(val=0, der=1.0)
    """
    if (x.val >= 1) or (x.val < -1):
        raise ValueError("Arccosine undefined at x=1, values greater than 1 or less than -1")
    newX = Var(np.arccos(x.val), derivative = -1/np.sqrt(1-x.val**2)*x.der)
    return newX

def arctan(x):
    """Returns a new Var with arctangent applied to the input Var x
    
    :param x: object on which arctangent is applied, required
    :type x: AD_Object.Var
    :return: new object with arctangent applied to input
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_BasicMath import cos
    >>> from src.autodiff.AD_Object import Var
    >>> arctan(Var(0))
    Var(val=0, der=1.0)
    """
    newX = Var(np.arctan(x.val), der = 1/(1+x.val**2)*x.der)
    return newX