import jax
import numpy as onp
import pacopy
import scipy.linalg as sp
import numdifftools as nd

class ContProblem:
    def __init__(self, func, u0, lmbda0,
    cont_method='Euler-Newton',
    callback=None, 
    jac_mode='Forward',
    max_steps=10,
    newton_tol=1e-10):
        # number of elements in u
        self.u0 = u0
        self.lmbda0 = lmbda0
        self.n = len(u0)
        # func must have two arguments: u and lmbda
        self.func = func
        self.cont_method = cont_method
        self.jac_mode = jac_mode
        self.max_steps = max_steps
        self.newton_tol = newton_tol
        self.callback = callback
        # Set JAX mode
        if jac_mode == 'Forward' or jac_mode == 'Reverse':
            self.no_jax = False
        elif jac_mode == 'Complex' or jac_mode == 'Numerical':
            self.no_jax = True
        else:
            raise Exception('Unknown Jacobian mode specified')
        # Adjust format of u0
        self.u0 = onp.array(u0) if self.no_jax else jax.numpy.array(u0)
        self.df_dlmbda_vec = None
        # derivative only wrt u
        self.jac_mat = None
        # update parameter values for grad and jac
        self.prev_lmbda_for_grad = None
        self.prev_lmbda_for_jac = None
    
    def f(self, u, lmbda):
        if self.no_jax:
            return onp.array(self.func(u, lmbda))
        else:
            return jax.numpy.array(self.func(u, lmbda))

    def df_dlmbda(self, u, lmbda):
        if self.prev_lmbda_for_grad != lmbda:
            # compute jac and df_dlmbda
            self.update_dfdlmbda(u, lmbda)
    
        return self.df_dlmbda_vec

    def jac(self, u, lmbda):
        if self.prev_lmbda_for_jac != lmbda:
            # compute jac and df_dlmbda
            self.update_jac(u, lmbda)

        return self.jac_mat

    def jacobian_solver(self, u, lmbda, rhs):
        # solver might do in-place operations
        M = self.jac(u, lmbda).copy()
        if self.no_jax:
            return sp.solve(M, rhs)
        else:
            return jax.scipy\
                .linalg.solve(M, rhs)

    def inner(self, a, b):
        """The inner product of the problem. Can be numpy.dot(a, b), but factoring in
        the mesh width stays true to the PDE.
        """
        if self.no_jax:
            return onp.dot(a, b)
        else:
            return jax.numpy.dot(a, b)

    def norm2_r(self, a):
        """The norm in the range space; used to determine if a solution has been found.
        """
        if self.no_jax:
            return onp.dot(a, a)
        else:
            return jax.numpy.dot(a, a)

    def update_jac(self, u, lmbda):
        _f = lambda x: self.f(x, lmbda)
        if self.jac_mode == 'Forward':
            self.jac_mat = jax.jacfwd(_f)(u)
        elif self.jac_mode == 'Reverse':
            self.jac_mat = jax.jacrev(_f)(u)
        elif self.jac_mode == 'Numerical':
            self.jac_mat = nd.Jacobian(_f, method='central')(u)
        elif self.jac_mode == 'Complex':
            self.jac_mat = nd.Jacobian(_f, method='complex')(u)
        else:
            raise Exception('Invalid Autodiff mode specified')
        self.prev_lmbda_for_jac = lmbda

    def update_dfdlmbda(self, u, lmbda):
        _f = lambda x: self.f(u, x)
        if self.jac_mode == 'Forward':
            self.df_dlmbda_vec = jax.jacfwd(_f)(lmbda)
        elif self.jac_mode == 'Reverse':
            self.df_dlmbda_vec = jax.jacrev(_f)(lmbda)
        elif self.jac_mode == 'Numerical':
            self.df_dlmbda_vec = nd.Jacobian(_f, method='central')(lmbda)[0]
        elif self.jac_mode == 'Complex':
            self.df_dlmbda_vec = nd.Jacobian(_f, method='complex')(lmbda)[0]
        else:
            raise Exception('Invalid Autodiff mode specified')
        self.prev_lmbda_for_grad = lmbda

    def solve(self):
        if self.cont_method == 'Euler-Newton':
            pacopy.euler_newton(
                    self, 
                    self.u0, 
                    self.lmbda0, 
                    self.callback, 
                    max_steps=self.max_steps, 
                    newton_tol=self.newton_tol
                )
        elif self.cont_method == 'Natural':
            pacopy.natural(
                    self, 
                    self.u0, 
                    self.lmbda0, 
                    self.callback, 
                    max_steps=self.max_steps, 
                    newton_tol=self.newton_tol
                )
        



