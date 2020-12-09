import numpy as np


class Var:
    """Var class is the base class for a variable in this automatic 
    differentiation library. Called with a val of type int or float, 
    and optional kwargs, namely derivative. Defining derivative 
    overwrites the default seed derivative value of 1 when calling a 
    new Var type.
    
    :return: Var object with val and der attributes
    :rtype: AD_Object.Var
    
    :example:
    >>> from src.autodiff.AD_Object import Var
    >>> x = Var(1, derivative=2)
    >>> print(x)
    Var(val=1, der=2)
    >>> x**2 + 2*x + 1
    Var(val=4, der=6)
    """

    def __init__(self, val, **kwargs):
        self.val = val
        if 'derivative' in kwargs and (isinstance(kwargs['derivative'], int) or isinstance(kwargs['derivative'], float)):
            self.der = kwargs['derivative']
        else:
            self.der = 1
        self.args = kwargs

    def __repr__(self):
        return f"Var(val={self.val}, der={self.der})"

    def __add__(self, other):
        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val + other
                new_der = self.der
            else:
                raise ValueError(
                    "Please use a Var type or num type for operations on Var")
        return Var(new_val, derivative=new_der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            new_val = self.val - other.val
            new_der = self.der - other.der
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val - other
                new_der = self.der
            else:
                raise ValueError(
                    "Please use a Var type or num type for operations on Var")

        return Var(new_val, derivative=new_der)

    def __rsub__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise ValueError(
                "Please use a Var type or num type for operations on Var")
        return Var(other, derivative=0).__sub__(self)

    def __mul__(self, other):
        try:
            new_val = self.val * other.val
            new_der = self.der * other.val + self.val * other.der
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val * other
                new_der = self.der * other
            else:
                raise ValueError(
                    "Please use a Var type or num type for operations on Var")
        return Var(new_val, derivative=new_der)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):  # no div in Python, truediv
        try:
            new_val = self.val / other.val
            new_der = (self.der * other.val -
                       self.val * other.der)/other.val**2
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                try:
                    new_val = self.val / other
                    new_der = self.der / other
                except ZeroDivisionError:
                    raise ValueError("Cannot divide by 0")
            else:
                raise ValueError(
                    "Please use a Var type or num type for operations on Var")
        return Var(new_val, derivative=new_der)

    def __rtruediv__(self, other):
        if not (isinstance(other, int) or isinstance(other, float)):
            raise ValueError(
                "Please use a Var type or num type for operations on Var")
        return Var(other, derivative=0).__truediv__(self)

    def __neg__(self):
        return self.__mul__(-1)

    def __pow__(self, other):
        try:
            new_val = self.val ** other.val
            # applying exp rule
            # i.e. a^b = e^(b*log(a)) => a^b*((a'*b)/a + b'*log(a))
            if self.val == 0:
                raise ValueError("Derivative at 0 not found")
            else:
                new_der = new_val * \
                    (((self.der*other.val)/self.val) + other.der*np.log(self.val))
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val ** other
                new_der = other * (self.val ** (other - 1))
            else:
                raise ValueError(
                    "Please use a numtype or Var type for the power")
        return Var(new_val, derivative=new_der)

    def __rpow__(self, other):
        # Cover case in which other is invalid type
        if not (isinstance(other, int) or isinstance(other, float)):
            print("test")
            raise ValueError(
                "Please use a Var type or num type for operations on Var")
        return Var(other, derivative=0).__pow__(self)

    def __eq__(self, other):
        if (isinstance(other, int) or isinstance(other, float)):
            return self.der == 0 and self.val == other
        elif isinstance(other, Var):
            return self.der == other.der and self.val == other.val
        else:
            raise ValueError("Please use a Var type or num type for operations on Var")
    
    def __ne__(self, other):
        return not self.__eq__(other)