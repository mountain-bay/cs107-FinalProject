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

