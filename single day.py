import sympy as sp
import numpy as np
from sympy import lambdify
from scipy.integrate import quad


mu = 0.15
i = 1376 * 0.8
s = 20
t = sp.symbols('t')
theta = 0.874
delta = 0.409
beta = 2.433
phi = (np.pi / 2) + (15 / 180) * np.pi * t
h = sp.asin(sp.cos(theta) * sp.sin(delta) + sp.sin(theta) * sp.cos(delta))
alpha = sp.Abs(theta - h)


OA = sp.Matrix([sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)])
e = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
n = OA.cross(e)
l = sp.Matrix([0, -sp.cos(alpha), -sp.sin(alpha)])
a = e
b = sp.sin(beta)*OA - sp.cos(beta)*n
m = a.cross(b)
gamma = sp.acos(sp.Abs(l.dot(m))) - np.pi / 2
epsilon = sp.asin(sp.sin(gamma) / 1.5)
j = i * (((sp.sin(2 * gamma) * sp.sin(2 * epsilon)) / ((sp.sin(gamma + epsilon)) * (sp.sin(gamma + epsilon)))) + ((sp.sin(2 * gamma) * sp.sin(2 * epsilon) / ((sp.sin(gamma + epsilon)) * (sp.sin(gamma + epsilon)) * (sp.cos(gamma - epsilon))) * (sp.cos(gamma - epsilon))))) / 2
p = j * sp.cos(epsilon) * s
f = lambdify(t, p, 'numpy')
W, error = quad(f, (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180), (np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180))
W = W * 0.15


print('fixed board' + str(W))


chi = (15 * np.pi / 180) * t
k = sp.asin(sp.cos(theta) * sp.sin(delta) + sp.sin(theta) * sp.cos(delta) * sp.cos(chi))
zeta = k - beta + (np.pi / 2)
tau = sp.asin(sp.sin(zeta) / 1.5)
o = i * (((sp.sin(2 * zeta) * sp.sin(2 * tau)) / ((sp.sin(zeta + tau)) * (sp.sin(zeta + tau)))) + ((sp.sin(2 * zeta) * sp.sin(2 * tau) / ((sp.sin(zeta + tau)) * (sp.sin(zeta + tau)) * (sp.cos(zeta - tau))) * (sp.cos(zeta - tau))))) / 2
q = o * sp.cos(tau) * s
r = lambdify(t, q, 'numpy')
V, error = quad(r, (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180), (np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180))
V = V * 0.15


print('single rotation axis' + str(V))


eta = i * 6 / 6.25 * s
iota = lambdify(t, eta, 'numpy')
U, error = quad(iota, (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180), (np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180))
U = U * 0.15


print('dual rotation axis' + str(U))