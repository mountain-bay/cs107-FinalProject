# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for autodiff.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

# import pytest
from src.autodiff.AD_Object import Var

try:
    x = Var(0, derivative=1)
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')
try:
    y = Var(1)
except NotImplementedError:
    assert 0 == 1, AssertionError('Var init not implemented')