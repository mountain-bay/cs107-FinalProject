class Var:
    """
    TODO: docs
    """
    def __init__(self, val, der = 1):
        self.val = val
        self.der = der
        
    def __repr__(self):
        return f"Var(val={self.val}, der={self.der})"
    
    def __add__(self, other):
        try:
            new_val = self.val + other.val
            new_der = self.der + other.der
        except:
            new_val = self.val + other
            new_der = self.der + other

        return Var(new_val, new_der)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            new_val = self.val - other.val
            new_der = self.der - other.der
        except:
            new_val = self.val - other
            new_der = self.der - other
            
        return Var(new_val, new_der)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        try:
            new_val = self.val * other.val
            new_der = self.der * other.val + self.val * other.der
        except:
            new_val = self.val * other
            new_der = self.der * other

        return Var(new_val, new_der)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        return self.__div__(other)
    
    def __neg__(self):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError
