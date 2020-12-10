## Usage instructions

Install package from TestPyPI:

`pip install -i https://test.pypi.org/simple/ mountainbay-autodiff`

Now that it's installed, you have access to the Var object and the functions exp, log, ln, sqrt, sin, cos, tan

```
import autodiff as AD

x = AD.Var(5)
# The above could be considered f(x) = x at x = 5
x.val # This is the given value 1
x.der # This is the default derivative, 1
```

We can operate on this object as a variable and with AD namespace functions to create complicated functions

```
y = AD.exp(x) + x**2 + x/10 - 5*x + 3
z = AD.sin(y)**AD.cos(x)
# We get new Var objects with updated values and derivatives from x = 5
```

You can get the value and derivative of various functions at any given point using these methods