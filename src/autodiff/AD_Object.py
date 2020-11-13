
class Var:
    """
    TODO: docs
    """
    def __init__(self, val, **kwargs):
        self.val = val
        if 'derivative' in kwargs:
            self.der = kwargs['derivative']
        else:
            self.der = 1
        self.args = kwargs
        
    def __repr__(self):
        return f"Var(val={self.val}, der={self.der})"

    def __add__(self, other):
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __div__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError
    
    def __neg__(self):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError