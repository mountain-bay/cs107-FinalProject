# Mountain-Bay Automatic Differentiation
- [Introduction](#introduction)
- [Background](#background)
- [How-To](#how-to-use-the-package)
- [Project Organization](#organization)
- [Implementation](#implementation)

## Introduction - Nicole
_what the problem solves and why it's important_
### Motivation
There are presently only two common methods for computing derivatives

    - Symbolic Derivatives
        - Computationally heavy
        - Exact number
    - Finite Difference
        - linear approximation
        - lacks machine precision

Automatic Differentiation, finding roots, is crucial to optimization and can be applied in many fields including machine learning, computer vision, and AI

## Math Background
**Automatic Differentiation (AD)** is a method of finding the extrema of functions, This could be for optimization problems, to find the local maxima, or for minimization to find the roots. The key to AD is breaking down complicated functions to much simpler/more manageable functions using the major components that follow 
### Major Components
#### Chain Rule
The chain rule is as follows: 

For every function that can be defined as a composite of functions, one function acting on another but with respect to the same variable, here _t_:

![Chain Rule Logic](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Cforall%20f%20%5Ctextrm%7B%20s.t.%20%7Df%28t%29%20%3D%20x%28y%28t%29%29)

The derivative of the composite function _f_ can be found by applying the derivative on the outer function _x_ with respect to _t_, and multiplying by the derivative of the inner function _y_ with respect to _t_:

![Chain Rule Definition](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20t%7D%5Ccdot%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20t%7D)

This can be applied any number of times (![Chain Rule More](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csmall%20%5Cfrac%20%7B%5Cpartial%20x%28y%28z%28t%29%29%29%29%7D%7B%5Cpartial%20t%7D%20%3D%20%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20t%7D%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20t%7D%5Cfrac%7B%5Cpartial%20z%7D%7B%5Cpartial%20t%7D)), and is the core of automatic differentiation.

AD uses the chain rule to <>
#### The Gradient
When the variable isn't **scalar** (one dimensional) like the above, it is a **vector**. This comes from linear algebra and just means that we're working in multiple dimensions now.
What changes is not the function but the base variable. The _t_ from above becomes: 
![Gradient Base Variable](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20x%20%5Cin%20%5Cmathbb%7BR%7D%5Em%20%3D%20%5Cbegin%7Bbmatrix%7D%20x_1%20%5C%5C%20x_2%20%5C%5C%20%5Cvdots%20%5Cend%7Bbmatrix%7D)

To obtain the derivative of the composite function, we use what is called a _gradient_ denoted by ![Upside-down Triangle](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Ctriangledown) which contains all the partial derivatives of the function across the vector as so:

![Gradient definition](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Ctriangledown%20f%28x%20%29%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_1%7D%5C%5C%20%5C%5C%20%5Cfrac%7B%5Cpartial%20f%7D%7B%5Cpartial%20x_2%7D%5C%5C%20%5C%5C%20%5Cvdots%20%5Cend%7Bbmatrix%7D)

The chain rule applies here as well, so the gradient with respect to a vector _x_ of a given composite function _h = h(u(x))_ is the partial derivative of _h_ with respect to _u_ times the gradient of _u_

![Gradient Chain Rule](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Ctriangledown_x%20h%28u%28x%29%29%3D%5Cfrac%7B%5Cpartial%20h%7D%7B%5Cpartial%20u%7D%20%5Ccdot%20%5Ctriangledown%20u%20%5Ctextrm%7B%20where%20%7D%20%5Ctriangledown%20u%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20x%7D%20%5C%5C%20%5C%5C%20%5Cfrac%7B%5Cpartial%20u%7D%7B%5Cpartial%20y%7D%20%5C%5C%20%5C%5C%20%5Cvdots%20%5Cend%7Bbmatrix%7D)

This can again be applied many times over on functions composed of composite functions, etc.

#### Elementary Functions
These are the base functions that have known (and therefore easy) derivatives, and from these functions all others are composed

Examples of these are: log, sin, polynomial, exp, etc

#### Evaluation Trace

Breaking down each function into its elementary functions can be called an evaluation trace, moving from inside out. 

It can be difficult to evaluate a composite function at any given value, so an evaluation trace follows the steps, evaluating the composite function at a given value from the most internal function moving out, so the most internal function is evaluated at the given input, and the function acting on that evaluates on the output, and so forth until the whole composite function is evaluated. Here's an example:

![Example Evaulation Trace](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Ctextrm%7BLet%20%7D%20f%20%3D%20%5Cexp%28%5Csin%28x%5E2%29%29%5C%5C%20f%28%5Csqrt%7B%5Cpi%7D%29%5Ctextrm%7B%20can%20be%20found%20using%20intermediate%20values%20%7D%20v_i)

| Trace | Operation | (_value_) |
| --- | --- | --- |
| ![x_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20x_1) | x | ![sqrtpi](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csqrt%7B%5Cpi%7D) |
| ![v_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_1) | ![squared](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28x_1%29%5E2) | ![pi](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cpi) |
| ![v_2](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_2) | ![sin](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csin%28v_1%29) | 0 |
| _f_ | ![exp](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cexp%28v_2%29) | 1 |

### Embedded Derivative

The key to AD forward evaluation is the Evaluation Trace. This is Automatic Differentiation.

| Trace | Operation | Derivative | (_value, derivative_) |
| --- | --- | --- | --- |
| ![x_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20x_1) | x | 1 | ![x1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28%5Csqrt%7B%5Cpi%7D%2C1%29) |
| ![v_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_1) | ![squared](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28x_1%29%5E2) | ![deriv1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%202%5Ccdot%20%5Cdot%7Bx_1%7D) | ![v1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28%5Cpi%2C2%29) |
| ![v_2](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_2) | ![sin](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csin%28v_1%29) | ![sinderiv](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Ccos%28v_1%29%5Ccdot%20%5Cdot%7Bv_1%7D) | ![02](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%280%2C2%29) |
| _f_ | ![exp](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cexp%28v_2%29) | ![expderiv](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20e%5E%7Bv_2%7D%5Ccdot%20%5Cdot%7Bv_2%7D) | ![12](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%281%2C2%29) |

With the derivative embedded as we trace the evaluation of the function, we have an automatic differentiation process in the forward evaluation method. 

There is more to say to follow the reverse method, but for now we leave it.

## Usage instructions
Install package from PyPI:

`pip3 install <autodiff> # Package name to be decided on`

Once installed, you can instantiate functions used in the differentiation process.

This might look something along the lines of:

```
import autodiff as ad # Package name to be decided on
import numpy as np    # Base our variables on numpy

def f(x):             # Define a function
    y = np.pow(x, 2)  # y = x^2
    return y

f_prime = ad.grad(f)  # Calculate gradient function
print(f_prime(1))     # Calculate gradient at x = 1
```

Output of above code segment would be `2` (f'(x) = 2x and f'(1) = 2). 

The primary AD object will be `autodiff.grad`, as that serves as the backbone of autodifferentiation. More objects may be added as we have a clearer idea of implementation details (e.g. specifying forward v. reverse modes, returning Jacobians, etc).

## Organization - Ninon

    What will the directory structure look like?
        - MountainBay/
            - __init__.py
            - AD_Object/
                - __init__.py
                - python.py
           - AD_BasicMath/
                - __init__.py
                - python.py
            - jacobian/
                - __init__.py
                - python.py
            - forward/
                - __init__.py
                - python.py
            - reverse/
                - __init__.py
                - python.py
            - mixed/
                - __init__.py
                - python.py

    What modules do you plan on including? What is their basic functionality?
        - AD_Object
            This will instantiate an Automatic Differentiation Object to be used in a forward or reverse mode, using the function and value given as input. It will contain methods that would calculate the Jacobian matrix, as well as AD in forward and reverse mode.
        -AD_BasicMath
            This module will contain basic operations, such as addition, substraction, and multiplication to be used on an AD object. It will also contain exponential and trig functions such as sin, cos, tan.
        
    Where will your test suite live? Will you use TravisCI? CodeCov?
        Since we have already integrated TravisCI and CodeCov, our test suite will live there.
    How will you distribute your package (e.g. PyPI)?
        This package will be distributed on PyPI, which allows users to upload packages.
    How will you package your software? 
        We will use wheel and setuptools in order to generate distribution packages for our package. 
    Will you use a framework? If so, which one and why? If not, why not?
        We will be using PyScaffold, because it sets up a folder system for us and incorporates Sphinx, which builds documentation.
        
## Implementation
Core data structures:

    The core data structures we anticipate using are matrices (ex. Jacobian), vectors (ex. seed vector), lists, tuples, and/or dictionaries for storing information.

What classes will you implement?

    We will be implementing a class that takes in a derivative and a value as input, and outputs an object for every calculation.
    We will also be implementing a class containing basic arithmetic operations.

What method and name attributes will your classes have?

    The class that takes in the derivative and value (AD_Object) will have a variety of methods which would conduct basic calculations, as well as the following: 
             - Jacobian
            This will calculate the Jacobian matrix for the given AD Object.
            - Forward
            This will calculate AD in forward mode.
            - Reverse
            This will calculate AD in reverse mode.
            - Mixed
            This will calculate AD in mixed mode.
    The class that contains basic arithmetic operations (AD_BasicMath) would contain methods to sum, subtract, multiply, divide, exponential, and trig functions as well.

What external dependencies will you rely on?

    We will be using numpy because of its mathematical capabilities and pandas, since it has an easy to use interface and fast data structures..
    
How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?


