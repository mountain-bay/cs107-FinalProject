# import numpy as np
from src.autodiff.AD_Object import Var

try:
    x = Var(0, derivative=1)
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')
try:
    y = Var(1)
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')

def test_x_Var_init():
    assert x.val == 0, AssertionError('Var init val fail')
    assert x.der == 1, AssertionError('Var init der fail')
    assert isinstance(x.args, dict), AssertionError('Var init kwargs fail')
def test_y_Var_init():
    assert y.val == 1, AssertionError('Var init w/o der val fail')
    assert y.der == 1, AssertionError('Var init w/o derivative der fail')
    assert isinstance(y.args, dict), AssertionError('Var init kwargs fail')
def test_repr():
    assert repr(x) == "Var(val=0, der=1)"
    assert repr(y) == "Var(val=1, der=1)"
def test_add():
    numadd = x + 2
    fltadd = x + 1.0
    varadd = x + y
    assert numadd.val == 2, AssertionError('Add num val fail')
    assert numadd.der == 1, AssertionError('Add num der fail')
    assert fltadd.val == 1, AssertionError('Add flt val fail')
    assert fltadd.der == 1, AssertionError('Add flt der fail')
    assert varadd.val == (x.val + y.val), AssertionError('Add var val fail')
    assert varadd.der == (x.der + y.der), AssertionError('Add var der fail')