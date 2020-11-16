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

def test_radd():
    numadd = 2 + x
    fltadd = 1.0 + x
    varadd = y + x
    assert numadd.val == 2, AssertionError('rAdd num val fail')
    assert numadd.der == 1, AssertionError('rAdd num der fail')
    assert fltadd.val == 1, AssertionError('rAdd flt val fail')
    assert fltadd.der == 1, AssertionError('rAdd flt der fail')
    assert varadd.val == (x.val + y.val), AssertionError('rAdd var val fail')
    assert varadd.der == (x.der + y.der), AssertionError('rAdd var der fail')

def test_sub():
    numsub = x - 2
    fltsub = x - 1.0
    varsub = x - y
    assert numsub.val == -2, AssertionError('sub num val fail')
    assert numsub.der == 1, AssertionError('sub num der fail')
    assert fltsub.val == -1, AssertionError('sub flt val fail')
    assert fltsub.der == 1, AssertionError('sub flt der fail')
    assert varsub.val == (x.val - y.val), AssertionError('sub var val fail')
    assert varsub.der == (x.der - y.der), AssertionError('sub var der fail')

def test_rsub():
    numsub = 2 - x
    fltsub = 1.0 - x
    varsub = y - x
    assert numsub.val == -2, AssertionError('rSub num val fail')
    assert numsub.der == 1, AssertionError('rSub num der fail')
    assert fltsub.val == -1, AssertionError('rSub flt val fail')
    assert fltsub.der == 1, AssertionError('rSub flt der fail')
    assert varsub.val == (y.val - x.val), AssertionError('rSub var val fail')
    assert varsub.der == (y.der - x.der), AssertionError('rSub var der fail')

def test_mul():
    nummul = x * 2
    fltmul = x * 1.0
    varmul = x * y
    assert nummul.val == 0, AssertionError('mul num val fail')
    assert nummul.der == 2, AssertionError('mul num der fail')
    assert fltmul.val == 0, AssertionError('mul flt val fail')
    assert fltmul.der == 1, AssertionError('mul flt der fail')
    assert varmul.val == (x.val * y.val), AssertionError('mul var val fail')
    assert varmul.der == (x.der * y.der), AssertionError('mul var der fail')

def test_rmul():
    nummul = 2 * x
    fltmul = 1.0 * x
    varmul = y * x
    assert nummul.val == 0, AssertionError('rmul num val fail')
    assert nummul.der == 2, AssertionError('rmul num der fail')
    assert fltmul.val == 0, AssertionError('rmul flt val fail')
    assert fltmul.der == 1, AssertionError('rmul flt der fail')
    assert varmul.val == (y.val * x.val), AssertionError('rmul var val fail')
    assert varmul.der == (y.der * x.der), AssertionError('rmul var der fail')

def test_operation_checks():
    # try bad add
    try:
        failadd = x + 'str'
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"failadd wrong exception {e} fail")
    else:
        raise AssertionError("failadd no exception fail")
    # try bad sub
    try:
        failsub = x - 'str'
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"failadd wrong exception {e} fail")
    else:
        raise AssertionError("failadd no exception fail")
    # try bad mul
    try:
        failmul = x * 'str'
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"failadd wrong exception {e} fail")
    else:
        raise AssertionError("failadd no exception fail")
    # try bad der 
    failder = Var(1, derivative='foo')
    assert(failder.val == 1) 
    assert(failder.der == 1)
    assert(failder.args['derivative'] == 'foo')