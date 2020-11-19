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
try:
    newx = Var(2)
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
    numsub = 2-x
    fltsub = 1.0-x
    varsub = y - x
    assert numsub.val == 2, AssertionError('rSub num val fail')
    assert numsub.der == -1, AssertionError('rSub num der fail')
    assert fltsub.val == 1, AssertionError('rSub flt val fail')
    assert fltsub.der == -1, AssertionError('rSub flt der fail')
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


def test_truediv():
    nummul = x / 2
    fltmul = x / 1.0
    varmul = x / y
    assert nummul.val == 0, AssertionError('truediv num val fail')
    assert nummul.der == 0.5, AssertionError('truediv num der fail')
    assert fltmul.val == 0, AssertionError('truediv flt val fail')
    assert fltmul.der == 1, AssertionError('truediv flt der fail')
    assert varmul.val == (
        x.val / y.val), AssertionError('truediv var val fail')
    assert varmul.der == (
        x.der / y.der), AssertionError('truediv var der fail')


def test_rtruediv():
    nummul = 2 / y  # x = 0, so switching order of x and y here to rly test it out
    fltmul = 1.0 / y
    varmul = x / y
    assert nummul.val == 2, AssertionError('rtruediv num val fail')
    assert nummul.der == -2, AssertionError('rtruediv num der fail')
    assert fltmul.val == 1, AssertionError('rtruediv flt val fail')
    assert fltmul.der == -1, AssertionError('rtruediv flt der fail')
    assert varmul.val == (
        x.val / y.val), AssertionError('rtruediv var val fail')
    assert varmul.der == (
        x.der / y.der), AssertionError('rtruediv var der fail')


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
        raise AssertionError(f"failsub wrong exception {e} fail")
    else:
        raise AssertionError("failsub no exception fail")
    # try bad rsub
    try:
        failsub = 'str' - x
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"failrsub wrong exception {e} fail")
    else:
        raise AssertionError("failrsub no exception fail")
    # try bad mul
    try:
        failmul = x * 'str'
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"failmul wrong exception {e} fail")
    else:
        raise AssertionError("failmul no exception fail")
     # try bad div
    try:
        faildiv = x / 'str'
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"faildiv wrong exception {e} fail")
    # try bad rtruediv
    try:
        faildiv = x.__rtruediv__('str')
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"faildiv wrong exception {e} fail")
     # try bad div 0
    try:
        faildiv = x / 0
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"faildiv wrong exception {e} fail")
     # try bad rdiv 0
    try:
        faildiv = 1 / x
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"faildiv wrong exception {e} fail")
    # try bad der
    failder = Var(1, derivative='foo')
    assert(failder.val == 1)
    assert(failder.der == 1)
    assert(failder.args['derivative'] == 'foo')


def test_pow_num():
    pow3 = newx**3
    pow2 = y**2
    pow0 = newx**0
    assert pow3.val == 2**3, AssertionError("2**3 val fail")
    assert pow3.der == 12, AssertionError("2**3 der fail")
    assert pow2.val == 1, AssertionError("1**2 val fail")
    assert pow2.der == 2, AssertionError("1**2 der fail")
    assert pow0.val == 1, AssertionError("2**0 val fail")
    assert pow0.der == 0, AssertionError("2**0 der fail")


def test_pow_var():
    xy = newx**y
    yx = y**newx
    assert xy.val == 2 ** 1, AssertionError("var pow val fail")
    assert xy.der > 2 and xy.der < 3, AssertionError("var pow der fail")
    assert yx.val == 1, AssertionError("var pow val fail")
    assert yx.der == 2, AssertionError("var pow der fail")


def test_pow_fail():
    # 0^x der fail
    try:
        x**y
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"bad der exception {e}")
    else:
        raise AssertionError("bad der fail")

    # Type checking
    try:
        y**('hello')
    except ValueError:
        pass
    except Exception as e:
        raise AssertionError(f"bad type exception {e}")
    else:
        raise AssertionError("bad type fail")


def test_neg():
    assert -x.val == 0, AssertionError("-x fail")
    assert -x.der == -1, AssertionError("-x fail")
    assert (-x*y).val == 0, AssertionError("-x*y fail")
    assert (-x*y).der == -1, AssertionError("-x*y fail")
    assert -y.val == -1, AssertionError("-y fail")
    assert -y.der == -1, AssertionError("-y fail")


def test_multiple_operations():
    fy = y**2 + 2*y + 5
    assert fy.val == 1**2 + 2 + \
        5, AssertionError("Operations combined val fail")
    assert fy.der == (2*1) + 2, AssertionError("Operations combined der fail")

