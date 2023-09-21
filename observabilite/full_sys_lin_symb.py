import numpy as np
import pandas as pd
import casadi as cas
import python_anesthesia_simulator as pas
import scipy.linalg as la
import sympy as sym


def jacobian(f, x):
    n = len(x)
    J = sym.zeros(1, n)
    for i in range(n):
        J[0, i] = f.diff(x[i])
    return J


# %% Test on full system
# %% Define the system
kp10, kp12, kp13, kp21, kp31, kpe = sym.symbols('kp10 kp12 kp13 kp21 kp31 kpe')
kr10, kr12, kr21, kr13, kr31, kre = sym.symbols('kr10 kr12 kr21 kr13 kr31 kre')

x = sym.symarray('x', 11)

u = sym.symarray('u', 2)

f = sym.Matrix([-(kp10 + kp12 + kp13) * x[0] + kp12 * x[1] + kp13 * x[2] + u[0],
                -kp21 * x[1] + kp21 * x[0],
                -kp31 * x[2] + kp31 * x[0],
                -kpe * x[3] + kpe * x[0],
                -(kr10 + kr12 + kr13) * x[4] + kr12 * x[5] + kr13 * x[6] + u[1],
                -kr21 * x[5] + kr21 * x[4],
                -kr31 * x[6] + kr31 * x[4],
                -kre * x[7] + kre * x[4],
                0,
                0,
                0])

h = x[3] * x[8] + x[7] * x[9] + x[10]

# get z expression

z = sym.symarray('z', 11)
z[0] = x[3] * x[8] + x[7] * x[9] + x[10]
# flag = True
# i = 1
# while flag:
#     phi = 

z[8] = x[8]
z[9] = x[9]
z[10] = x[10]
for i in range(1, 8):
    z[i] = np.sum(jacobian(z[i-1], x) @ f)


zu = sym.Matrix(z)
replace_dict = {u[0]: 0, u[1]: 0}
z = zu.subs(replace_dict)
print(z)
phi_u = zu - z
print(phi_u)

#%% test the bijectivity of the map
z_solve = sym.symarray('z', 11)

equation = [z_solve[i] - z[i] for i in range(11)]

sol = sym.solve(equation, [x[i] for i in range(11)])
print(sol)

#%% get z dynamics

z8dot = np.sum(jacobian(z[2], x) @ f)

z81, z82, z83, z84, z85, z86, z87, z88, z89, z810, z811 = sym.symbols('z81 z82 z83 z84 z85 z86 z87 z88 z89 z810 z811')

res = sym.solve([z8dot - z81*z[0] - z82*z[1] - z83*z[2] - z84*z[3] - z85*z[4] - z86*z[5] - z89*z[8] - z810*z[9] - z811*z[10]], z81, z82, z83, z84, z85, z86, z87, z88, z89, z810, z811, dict=True)

print(res)
z81_coeff = res[0][z81]
replace_dict = {z82: 0, z83: 0, z84: 0, z85: 0, z86: 0, z87: 0, z88: 0, z89: 0, z810: 0}
z81_coeff = z81_coeff.subs(replace_dict)
print(z81_coeff)