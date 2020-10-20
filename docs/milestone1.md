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

## Background - Zach
<!-- TODO -->

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
            - jacobian/
                - __init__.py
                - python.py
            - chain/
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
        - Jacobian
        - Chain Rule
        - Forward
        - Reverse
        - Mixed
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

What method and name attributes will your classes have?
    The class that takes in the derivative and value will have a variety of methods which would conduct basic calculations.

What external dependencies will you rely on?
    We will be using numpy because of its mathematical capabilities and pandas, since it has an easy to use interface and fast data structures..
    
How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)?


