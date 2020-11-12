# import numpy as np
from src.autodiff.AD_Object import Var

def test_Var_init():
    try:
        x = Var(0, derivative=1)
    except NotImplementedError:
        assert 0 == 1, AssertionError('Var init not implemented')

    assert x.val == 0, AssertionError('Var init val fail')
    assert x.der == 1, AssertionError('Var init derivative fail')
    assert isinstance(x.args, dict), AssertionError('Var init kwargs fail')
