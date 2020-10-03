import sys
import foobar



def test_foo_1():
    assert( foobar.inc(1) == 2 )
    assert( foobar.inc(1.0) == 2.0 )

def test_foo_neg():
    assert( foobar.inc(-1) == 0 )

def test_foo_big():
    assert( foobar.inc(sys.maxsize) == 9223372036854775808 )