import numpy as np
from src.autodiff.AD_Object import Var
from src.autodiff.AD_BasicMath import sin, cos, tan, ln, log, sqrt, exp, csc, sec, cot, arcsin, arccos, arctan

try:
    x = Var(0, derivative=1)
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
    assert sinx.val == 0, AssertionError('Sin val at 0 fail')
    assert sinx.der == 1, AssertionError('Sin der at 0 fail')

def test_cos():
    try:
       cosx = cos(x)
    except Exception as e:
        raise AssertionError(e)
    assert cosx.val == 1, AssertionError('cos val at 0 fail')
    assert cosx.der == 0, AssertionError('cos der at 0 fail')

def test_tan():
    try:
       tanx = tan(x)
    except Exception as e:
        raise AssertionError(e)
    assert tanx.val == 0, AssertionError('tan val at 0 fail')
    assert tanx.der == 1, AssertionError('tan der at 0 fail')
    try:
       tanx = tan(y)
    except Exception as e:
        raise AssertionError(e)
    assert tanx.val == np.tan(np.pi), AssertionError('tan val at pi fail')
    assert tanx.der == 1/(np.cos(np.pi)**2), AssertionError('tan der at pi fail')

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

def test_ln_undef():
    new = Var(-3)
    try:
        ln_new = ln(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_log():
    try:
       logx = log(x)
    except Exception as e:
        print(str(e) + 'log val at 0 fail')

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
    assert sqrt(x).val == 0, AssertionError('Sqrt 0 val fail')
    assert sqrt(x).der == 1, AssertionError('Sqrt 1 der fail')
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
    assert exp0.val == 1, AssertionError("exp(0) val fail")
    assert exp0.der == 1, AssertionError("exp(0) der fail")
    exp1 = exp(Var(1))
    assert exp1.val == np.e, AssertionError("exp(1) val fail")
    assert exp1.der == np.e, AssertionError("exp(1) der fail")


def test_csc():
    try:
       cscx = csc(x)
    except Exception as e:
        print(str(e))

def test_csc_undef():
    new = Var(np.pi)
    try:
        csc_new = csc(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_sec():
    try:
       secx = sec(x)
    except Exception as e:
        print(str(e))

def test_sec_undef():
    new = Var(np.pi/2)
    try:
        sec_new = sec(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_cot():
    try:
       cotx = cot(x)
    except Exception as e:
        print(str(e))

def test_cot_undef():
    new = Var(np.pi)
    try:
        cot_new = cot(new)
    except Exception as e:
        assert(isinstance(e, ValueError))
    
def test_arcsin():
    try:
       arcsinx = arcsin(x)
    except Exception as e:
        print(str(e))

def test_arcsin_undef():
    new = Var(np.pi)
    try:
        arcsin_new = arcsin(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_arccos():
    try:
       arccosx = arccos(x)
    except Exception as e:
        print(str(e))

def test_arccos_undef():
    new = Var(np.pi)
    try:
        arccos_new = arccos(new)
    except Exception as e:
        assert(isinstance(e, ValueError))

def test_arctan():
    try:
       arctanx = arctan(x)
    except Exception as e:
        print(str(e))

