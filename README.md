# FContin

![Made with Love in India](https://madewithlove.org.in/badge.svg)

Solve F(**u**, λ) = 0 over λ with just F!

This repository contains [natural](https://en.wikipedia.org/wiki/Numerical_continuation#Natural_parameter_continuation) and [pseudo-arclength/Euler-Newton](https://en.wikipedia.org/wiki/Numerical_continuation#Pseudo-arclength_continuation) continuation library using [JAX](https://github.com/google/jax) for automatic differentiation, [numdifftools](https://pypi.org/project/numdifftools/) for numerical differentiation using real or complex derivatives, and [Pacopy](https://github.com/nschloe/pacopy)

This enables automatic/numerical differentiation to obtain the Jacobian and derivative with respect to the parameter. GPU/TPU support is packaged as part of JAX.

# How to install and execute?

Just run 
```
pip install fcontin
```

The following program illustrates a basic example
```python
from fcontin.ContProblem import ContProblem

###
# Define problem
###

def f(u, lmbda):
    """The evaluation of the function to be solved
    """
    return [
        u[0] + u[1] - (lmbda + 1.), u[0] - u[1] - lmbda
    ]

###
# Solving and Plotting
###

# Initial guess
u0 = [0., 0.]
# Initial parameter value
lmbda0 = 1.0

# Creating the problem
# Natural or Euler-Newton for cont_method
# Forward, Reverse, Numerical, Complex for jac_mode
problem = ContProblem(f, u0, lmbda0,
cont_method='Euler-Newton', 
jac_mode='Complex',
max_steps=10,
newton_tol=1e-10,
callback=callback
)

problem.solve()
```

## Whom to contact?

Please direct your queries to [gpavanb1](http://github.com/gpavanb1)
for any questions.