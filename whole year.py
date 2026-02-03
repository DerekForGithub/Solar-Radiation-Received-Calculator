import sympy as sp
import numpy as np
from sympy import lambdify
from scipy.integrate import quad


def no_axis():
    global rho
    rho = 0
    for n in range(1, 365):
        isc = 1376
        #i = isc * (1 + 0.033 * sp.cos(2 * np.pi * n / 365))
        #i = 1
        i = 0.8 * isc
        s = 20
        t = sp.symbols('t')
        theta = 0.874
        delta = 0.4091 * np.sin(2 * np.pi * (284 + n) / 365)
        beta = 2.433
        phi = (np.pi / 2) + (15 / 180) * np.pi * t
        h = sp.asin(sp.cos(theta) * sp.sin(delta) + sp.sin(theta) * sp.cos(delta))
        alpha = sp.Abs(theta - h)
        upper_limit = (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180)
        lower_limit = -upper_limit

        OA = sp.Matrix([sp.sin(theta) * sp.cos(phi), sp.sin(theta) * sp.sin(phi), sp.cos(theta)])
        e = sp.Matrix([-sp.sin(phi), sp.cos(phi), 0])
        north = OA.cross(e)
        l = sp.Matrix([0, -sp.cos(alpha), -sp.sin(alpha)])
        a = e
        b = sp.sin(beta) * OA - sp.cos(beta) * north
        m = a.cross(b)
        gamma = sp.acos(sp.Abs(l.dot(m))) - np.pi / 2
        epsilon = sp.asin(sp.sin(gamma) / 1.5)
        j = i * (((sp.sin(2 * gamma) * sp.sin(2 * epsilon)) / (
                    (sp.sin(gamma + epsilon)) * (sp.sin(gamma + epsilon)))) + ((
                sp.sin(2 * gamma) * sp.sin(2 * epsilon) / (
                (sp.sin(gamma + epsilon)) * (sp.sin(gamma + epsilon)) * (sp.cos(gamma - epsilon))) * (
                    sp.cos(gamma - epsilon))))) / 2
        p = j * sp.cos(epsilon) * s
        f = lambdify(t, p, 'numpy')
        W, error = quad(f, upper_limit, lower_limit)
        W = W * 0.15
        rho = rho + W

    print('fixed board' + str(rho))


def one_axis():
    global psi
    psi = 0
    for n in range(1, 365):
        isc = 1376
        #i = isc * (1 + 0.033 * sp.cos(2 * np.pi * n / 365))
        #i = 1
        i = 0.8 * isc
        s = 20
        t = sp.symbols('t')
        theta = 0.874
        delta = 0.4091 * np.sin(2 * np.pi * (284 + n) / 365)
        beta = 2.433
        phi = (np.pi / 2) + (15 / 180) * np.pi * t
        h = sp.asin(sp.cos(theta) * sp.sin(delta) + sp.sin(theta) * sp.cos(delta))
        alpha = sp.Abs(theta - h)
        upper_limit = (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180)
        lower_limit = -upper_limit

        chi = (15 * np.pi / 180) * t
        k = sp.asin(sp.cos(theta) * sp.sin(delta) + sp.sin(theta) * sp.cos(delta) * sp.cos(chi))
        zeta = k - beta + (np.pi / 2)
        tau = sp.asin(sp.sin(zeta) / 1.5)
        o = i * (((sp.sin(2 * zeta) * sp.sin(2 * tau)) / ((sp.sin(zeta + tau)) * (sp.sin(zeta + tau)))) + ((
                sp.sin(2 * zeta) * sp.sin(2 * tau) / (
                (sp.sin(zeta + tau)) * (sp.sin(zeta + tau)) * (sp.cos(zeta - tau))) * (sp.cos(zeta - tau))))) / 2
        q = o * sp.cos(tau) * s
        r = lambdify(t, q, 'numpy')
        V, error = quad(r, upper_limit, lower_limit)
        V = V * 0.15
        psi = psi + V

    print('single rotation axis' + str(psi))


def two_axis():
    global pi
    pi = 0
    for n in range(1, 365):
        t = sp.symbols('t')
        s = 20
        isc = 1376
        #i = isc * (1 + 0.033 * sp.cos(2 * np.pi * n / 365))
        #i = 1
        i = 0.8 * isc
        theta = 0.874
        delta = 0.4091 * np.sin(2 * np.pi * (284 + n) / 365)
        upper_limit = (-np.arccos(-np.tan(delta) / np.tan(theta))) / (15 * np.pi / 180)
        lower_limit = -upper_limit

        eta = i * 6 / 6.25 * s
        iota = lambdify(t, eta, 'numpy')
        U, error = quad(iota, upper_limit, lower_limit)
        U = U * 0.15
        pi = pi + U

    print('dual rotation axis' + str(pi))


if __name__ == '__main__':
    main()