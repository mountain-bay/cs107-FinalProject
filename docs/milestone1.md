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

## How To Use the Package - Kaiwen
<!-- TODO -->

## Organization - Ninon

    What will the directory structure look like?
        - MountainBay/
            - __init__.py
            - jacobian/
                - __init__.py
                - python.py
            - chain/
                - __init__.py
                - python.py
            - foward/
                - __init__.py
                - python.py
            - reverse/
                - __init__.py
                - python.py
            - mixed/
                - __init__.py
                - python.py

    What modules do you plan on including? What is their basic functionality?
        - jacobian
        - chain
        - forward
        - extra module
    Where will your test suite live? Will you use TravisCI? CodeCov?
        We might as well keep them because they are integrated
    How will you distribute your package (e.g. PyPI)?
        PyPI is a perfect platform to distribute our package because it is widely available, and well documented (I imagine)
    How will you package your software? 
        pip-wheels
    Will you use a framework? If so, which one and why? If not, why not?
        We are considering other frameworks in the future
    Other considerations?

## Implementation - Everyone
How we're gonna make it
what data structures
    matrices and vectors
what classes
external dependencies
    - Numpy
        - mathematical POWER
    - Pandas
        - easy to use interface
        - fast data structures

