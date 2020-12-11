

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

Since the last Milestone, we have implemented reverse mode.

## Broader Impact

  Fortunately, there is not a very high chance our particular software will be misused, though it is not impossible. Perhaps one example of an application where our software could hypothetically be misused is if someone wanted to use reverse automatic differentiation in order to conduct backpropagation for their neural network, and this particular neural network was incredibly biased. For example, the Correctional Offender Management Profiling for Alternative Sanctions (COMPAS) tool was a machine learning model which exhibited racial bias and resulted in very real consequences, by labeling Black individuals more often than Whites as being high risk and therefore influencing judges in their sentencing. 
  
## Software Inclusivity

  In general, we are aware our software must inevitably be flawed in its inclusivity, not because of purposeful design, but rather because of the broader world we are situated in. One of the common approaches to improve inclusivity is to increase representation within tech. While this is a worthwhile goal, it is also a temporary fix, and also one that is not effective if women, non-native English speakers, LGBTQ+ folks, working parents, and/or people of color’s voices and critiques are not respected and amplified. The recent firing of well-known AI ethicist and Black woman Timnit Gebru for speaking up about the possible misuses of Natural Language Processing models  at Google illustrates this. Within our own group, varied in our identities and marginalization, we approached inclusivity by dividing tasks equally and having different team members provide feedback and approval for Pull Requests.


## Future Features


