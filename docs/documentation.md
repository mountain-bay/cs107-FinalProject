# Mountain-Bay Automatic Differentiation

- [Introduction](#introduction)
- [Background](#background)
- [How-To](#how-to-use-the-package)
- [Project Organization](#organization)
- [Implementation](#implementation)

## Introduction

_what the problem solves and why it's important_

### Motivation

Differentiation, or the process of finding the derivative of a function, is a cornerstone operation in computational science, with applications in many other scientific disciplines. While there are several ways to program differentiation, automatic differentiation (AD) is the most efficient (linear in the cost of computing the value), while being numerically stable.

The two common alternative methods to automatic differentiation are symbolic differentiation and finite difference. Though symbolic differentiation gives the exact number down to machine precision, it is computationally heavy and inefficient. Meanwhile, the finite difference method is a linear approximation, lacking machine precision, and is normally only used for testing. AD can handle complex functions while still returning accurate results.

Therefore, AD is a particularly useful tool in calculating derivatives, finding applications in fields as varied as mathematical optimization to machine learning and AI. Optimization utilizes the roots of an equation to maximize or minimize a function, a concept used widely from physics, biology, and engineering to economics and business. Finding derivatives through AD, mostly in the form of gradients and Hessians, are ubiquitous in machine learning, computer vision, and AI. Additional AD application include computational fluid dynamics, atmospheric sciences, and physical modeling.

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

![Example Evaluation Trace](https://latex.codecogs.com/png.latex?%5Cbg_white%20%5Cfn_jvn%20%5Clarge%20%5Ctextrm%7BLet%20%7D%20f%20%3D%20%5Cexp%28%5Csin%28x%5E2%29%29%5C%5C%20f%28%5Csqrt%7B%5Cpi%7D%29%5Ctextrm%7B%20can%20be%20found%20using%20intermediate%20values%20%7D%20v_i)

This function can be shown in a graph or as a trace table

![Example Computation graph](example_trace.png)

| Trace                                                                                  | Operation                                                                                            | (_value_)                                                                                                |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| ![x_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20x_1) | x                                                                                                    | ![sqrtpi](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csqrt%7B%5Cpi%7D) |
| ![v_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_1) | ![squared](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28x_1%29%5E2) | ![pi](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cpi)                  |
| ![v_2](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_2) | ![sin](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csin%28v_1%29)   | 0                                                                                                        |
| _f_                                                                                    | ![exp](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cexp%28v_2%29)   | 1                                                                                                        |

### Embedded Derivative

The key to AD forward evaluation is the Evaluation Trace. This is Automatic Differentiation.

| Trace                                                                                  | Operation                                                                                            | Derivative                                                                                                                       | (_value, derivative_)                                                                                          |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| ![x_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20x_1) | x                                                                                                    | 1                                                                                                                                | ![x1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28%5Csqrt%7B%5Cpi%7D%2C1%29) |
| ![v_1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_1) | ![squared](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28x_1%29%5E2) | ![deriv1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%202%5Ccdot%20%5Cdot%7Bx_1%7D)                 | ![v1](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%28%5Cpi%2C2%29)              |
| ![v_2](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20v_2) | ![sin](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Csin%28v_1%29)   | ![sinderiv](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Ccos%28v_1%29%5Ccdot%20%5Cdot%7Bv_1%7D) | ![02](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%280%2C2%29)                  |
| _f_                                                                                    | ![exp](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%5Cexp%28v_2%29)   | ![expderiv](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20e%5E%7Bv_2%7D%5Ccdot%20%5Cdot%7Bv_2%7D)   | ![12](https://latex.codecogs.com/png.latex?%5Cinline%20%5Cbg_white%20%5Cfn_jvn%20%281%2C2%29)                  |

With the derivative embedded as we trace the evaluation of the function, we have an automatic differentiation process in the forward evaluation method.


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

You can get the value and derivative of various functions at any given point using these methods.


## Organization

The directory structure for our project is as follows, illustrated in tree format. Note that our modules are stored within src/ad-project/ folder. We also anticipate storing documentation, examples, and our test suite in the appropriate folders, indicated by their names. Other than that, there are also the standard files we see in this type of repository: a README file, a license file, and a requirements file.
The structure:

`
MountainBay/

    -AUTHORS.rst
    -CHANGELOG.rst
    -docs/
        -_static/
        -authors.rst
        -changelog.rst
        -conf.py
        -index.rst
        -license.rst
        -Makefile
    -LICENSE.txt
    -README.rst
    -requirements.txt
    -setup.cfg
    -setup.py
    -src/
        -autodiff/
            -__init__.py
            -AD_Object.py
            -AD_BasicMath.py
        -AD_project.egg-info/
            -dependency_links.txt
            -not-zip-safe
            -PKG-INFO
            -requires.txt
            -SOURCES.txt
            -top_level.txt
    -tests/
        -conftest.py
        -test_AD_BasicMath.py
        -test_AD_Object.py

`
We have created two modules, AD_Object and AD_BasicMath. Our first module is called Var, and is stored under the module 'AD_Object'. This module instantiates an Automatic Differentiation (AD) Object to be used in a forward or reverse mode. It takes in a value and derivative, and returns our AD Object with new values and derivatives. In AD_Object, we perform operation overload for methods such as addition, subtraction, multiplication, division, power and negation. The second module is be AD_BasicMath. This module contains trigonometric functions such as sin, cos, tan, exponential functions, log and natural log functions. This allows us to carry out a variety of methods on our AD Object beyond the basic functions that we have overloaded.
We used the framework PyScaffold, because it sets up a folder system for us and incorporates Sphinx, which builds documentation. The directory structure set up by PyScaffold is illustrated above, with our own modules for AD_Object and AD_BasicMath included. We wrote out documentation and examples for each module. We also wrote tests, which live within the 'test' folder, and continued to use TravisCI and CodeCov.

## Implementation

The core data structures we anticipate used are vectors (ex. seed vector), lists, tuples, and/or dictionaries for storing information. The class AD_Object will creates our Auto Differentiation Object and has all operation overload for basic functions such as addition, subtraction, multiplication, division, and power. The class AD_Num contains elementary operations such as trigonometric functions, exponential functions, and more.

For external dependencies, used numpy because of its mathematical capabilities and pandas, since it has an easy to use interface and fast data structures.

## Our extension: Reverse Mode

The reverse mode is composed of two steps: a forward pass and a reverse pass. The forward pass evaluates the elementary functions and stores the partial derivative, but does not do the chain rule. The reverse pass begin with evaluating Vbar = (df/dVn) = 1, because f = Vn. It then works backwards to evaluate the derivatives, adding values when a node has multiple children (implementing the chain rule). Note: while the forward mode calculates the Jacobian-vector product, the reverse mode actually calculates the Jacobian transpose-product. 

We implemented Reverse mode by constructing a graph of nodes that represent our original expression as the program runs. This means an input x and/or y is the root of such a graph, and we think of Var as a node that is capable of having children. Whenever a new expression is built out of current nodes, that new expression would be a child of each of those current nodes. This also save contributing weights for gradient computation later on. In our ADObject module, we therefore extended our code by first initializing with an empty list, self.children, and reverse derivative/gradient value set equal to None, self.rder. For reverse mode, we are appending each new node as the child of the inputs that was saved into a list. We then propagate derivatives using recursion, using a function we have called "revder" within our ADObject module.

## Our extension: Vector Input

## Broader Impact

  Fortunately, there is not a very high chance our particular software will be misused, though it is not impossible. Perhaps one example of an application where our software could hypothetically be misused is if someone wanted to use reverse automatic differentiation in order to conduct backpropagation for their neural network, and this particular neural network was incredibly biased. For example, the Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) tool was a machine learning model which exhibited racial bias and resulted in very real consequences, by labeling Black individuals more often than Whites as being high risk and therefore influencing judges in their sentencing. 
  
## Software Inclusivity

  In general, we are aware our software must inevitably be flawed in its inclusivity, not because of purposeful design, but rather because of the broader world we are situated in. One of the common approaches to improve inclusivity is to increase representation within tech. While this is a worthwhile goal, it is also a temporary fix, and also one that is not effective if women, non-native English speakers, LGBTQ+ folks, working parents, and/or people of color’s voices and critiques are not respected and amplified. The recent firing of well-known AI ethicist and Black woman Timnit Gebru for speaking up about the possible misuses of Natural Language Processing models at Google illustrates this. Within our own group, varied in our identities and marginalization, we approached inclusivity by dividing tasks equally and having different team members provide feedback and approval for Pull Requests.


## Future Features
  A future feature and applied use of this program is for finding the extrema of an optimization problem. Optimization consists of selecting a "best element" from a set of alternatives. In the simplest case, this consists of maximizing or minimizing a function. 
  
  ### Gradient Descent/Ascent
  One method to find the min/max uses the gradient descent/ascent. Using the reverse mode of our program, you can get the gradient vector of a function at the specified point. Using the gradient we can then extend the program to do gradient descent/ascent to find a local minimum/maximum respectively. This extension would work in the following way. Given a function, an initial position, and the type of extrema desired (either min or max) the program would calculate the gradient at that point and then take a small step in either the negative or positive direction of the gradient (for finding minima or maxima respectively). It would repeat this process until the gradient is zero, meaning an extrema has been reached. If the step size is too big, we might expect the program to oscilate around an extreme point. Thus the implementation would optimally decrease the step size nearer to the extrema. Additionally, some threshold around gradient = 0 would be specified in detecting whether the point of extrema has been reached. 

  If a function has a bounded domain, the global extrema could be calculated by repeating the gradient descent/ascent algorithm over a sample space of the bounded domain with appropriately small step size, and then comparing the value of the function evaluatetd at every point of extrema. 

  ### Future Classes and Implementation
  In order to implement the gradient descent using our program, we would create a new class AD_GradExtrema(), which would take the function, an intial position, and a specificiation for min or max as inputs. Then, using the process outlined above, it would calculate a list of local max/min and return the greatest/smallest value in the list to find the global max/min. This returned value represents the best value for the optimization. 


