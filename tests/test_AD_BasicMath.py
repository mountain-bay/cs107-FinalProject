import numpy as np
from src.autodiff.AD_Object import Var
from src.autodiff.AD_BasicMath import sin, cos, tan, ln, log, sqrt, exp

try:
    x = Var([0,0], derivative=[1,1])
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')
try:
    y = Var(np.pi)
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')

def test_sin():
    try:
       sinx = sin(x)
    except Exception as e:
        raise AssertionError(e)
    assert (c == 0 for c in sinx.val), AssertionError('Sin val at 0 fail')
    assert (c == 1 for c in sinx.der), AssertionError('Sin der at 0 fail')

def test_cos():
    try:
       cosx = cos(x)
    except Exception as e:
        raise AssertionError(e)
    assert (c == 1 for c in cosx.val), AssertionError('cos val at 0 fail')
    assert (c == 0 for c in cosx.der), AssertionError('cos der at 0 fail')

def test_tan():
    try:
       tanx = tan(x)
    except Exception as e:
        raise AssertionError(e)
    assert (c == 0 for c in tanx.val), AssertionError('tan val at 0 fail')
    assert (c == 1 for c in tanx.der), AssertionError('tan der at 0 fail')
    try:
       tany = tan(y)
    except Exception as e:
        raise AssertionError(e)
    assert tany.val == np.tan(np.pi), AssertionError('tan val at pi fail')
    assert tany.der == 1/(np.cos(np.pi)**2), AssertionError('tan der at pi fail')

def test_tan_undef():
    new = Var(3*np.pi/2)
    try:
       tan_new = tan(new)
    except Exception as e:
        assert(isinstance(e, ValueError))
    try:
       tanx = tan(Var(1))
    except Exception as e:
        raise AssertionError(e)
    assert tanx.val == np.tan(1), AssertionError('tan defined between -pi/2, pi/2 fail')
    assert tanx.der == 1/(np.cos(1)**2), AssertionError('tan defined between -pi/2, pi/2 fail')

def test_ln():
    try:
       lnx = ln(x)
    except Exception as e:
        print(str(e) + 'ln val at 0 fail')
    # assert(lnx.val == 0)

def test_ln_undef():
    new = Var(-3)
    try:
        ln_new = ln(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_log():
    try:
       logx = log(Var(1))
    except Exception as e:
        print(str(e) + 'log val at 0 fail')
    assert(logx.val == 0)
    assert(logx.der == 1/np.log(10))
    try:
       logbase2 = log(Var(1), 2)
    except Exception as e:
        print(str(e) + 'log val at 0 fail')
    assert(logbase2.val == 0)
    assert(logbase2.der == 1/np.log(2))
    try:
       logbase3 = log(Var(3), 3)
    except Exception as e:
        print(str(e) + 'log val at 0 fail')
    assert(logbase3.val == 1)
    assert(logbase3.der == 1/(3*np.log(3)))
def test_log_undef():
    new = Var(-3)
    try:
        log_new = log(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_sqrt():
    square = Var(16)
    square_der = Var(4, derivative=16)
    newsq = sqrt(square)
    newsqder = sqrt(square_der)
    assert sqrt(Var(1)).val == 1, AssertionError('Sqrt 0 val fail')
    assert sqrt(Var(1)).der == 0.5, AssertionError('Sqrt 1 der fail')
    assert newsq.val == 4, AssertionError('Sqrt square val fail')
    assert newsq.der == 0.125, AssertionError('Sqrt square der fail')
    assert newsqder.val == 2, AssertionError('Sqrt square_der val fail')
    assert newsqder.der == 4, AssertionError('Sqrt square_der der fail')

def test_sqrt_undef():
    try:
        sqrt(Var(-1))
    except ValueError:
        pass
    except Exception:
        raise AssertionError('negative val root wrong exception')
    else:
        raise AssertionError('negative val root total fail')

def test_exp():
    exp0 = exp(x)
    assert (c == 1 for c in exp0.val), AssertionError("exp(0) val fail")
    assert (c == 1 for c in exp0.der), AssertionError("exp(0) der fail")
    exp1 = exp(Var(1))
    assert exp1.val == np.e, AssertionError("exp(1) val fail")
    assert exp1.der == np.e, AssertionError("exp(1) der fail")

