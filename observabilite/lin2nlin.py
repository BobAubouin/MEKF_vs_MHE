import numpy as np
import pandas as pd
import casadi as cas
import python_anesthesia_simulator as pas
import scipy.linalg as la
import sympy as sym

x = sym.symarray('x', 2)
C_50p, C_50r, gamma = sym.symbols('C_50p C_50r gamma')
a, b, c = sym.symbols('a b c')

U = x[0]/C_50p + x[1]/C_50r
h = 1 - U**gamma/(U**gamma + 1)

    
equation = [a - sym.diff(h, x[0]),
            b - sym.diff(h, x[1]),
            c - (h - sym.diff(h, x[0])*x[0] - sym.diff(h, x[1])*x[1])]

x_value = [4, 3]
a_value = sym.diff(h, x[0]).subs({x[0]: x_value[0], x[1]: x_value[1],
                                  C_50p: 4, C_50r: 19, gamma: 2}) + 0.0001
b_value = sym.diff(h, x[1]).subs({x[0]: x_value[0], x[1]: x_value[1],
                                  C_50p: 4, C_50r: 19, gamma: 2}) + 0.0001
c_value = (h - sym.diff(h, x[0])*x_value[0] - sym.diff(h, x[1])*x_value[1])
c_value = c_value.subs({x[0]: x_value[0], x[1]: x_value[1],
                        C_50p: 4, C_50r: 19, gamma: 2}) + 0.0001

equation = [equation[i].subs({a: a_value, b: b_value, c: c_value,
                              x[0]: x_value[0], x[1]: x_value[1], }) for i in range(3)]

sol = sym.nsolve(equation, [C_50p, C_50r, gamma], (2, 12, 8))

print(sol)

# %% other thing

x = sym.symarray('x', 5)
z = sym.symarray('z', 5)
k1, k2 = sym.symbols('k1 k2')
equation = [z[0] - x[0]*x[2] + x[1]*x[3] + x[4],
            z[1] - -k1*x[0]*x[2] - k2 * x[1]*x[3],
            z[2] - x[2],
            z[3] - x[3],
            z[4] - k1**2*x[0]*x[2] + k2**2*x[1]*x[3]]

sol = sym.solve(equation, [x[0], x[1], x[2], x[3], x[4]])
print(sol)
