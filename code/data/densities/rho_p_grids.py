#! /usr/bin/env python3

import math as math
import numpy as np
import numba as nb
import scipy.integrate as integrate

import matplotlib.pyplot as plt

pi2 = np.pi**2.
zeta3 = 1.202056903

num_pts = 10000
p_max = 3e2

x_grid = np.logspace(-3, np.log10(200), num_pts)
rho_boson_grid = np.zeros(num_pts)
rho_der_boson_grid = np.zeros(num_pts)
p_boson_grid = np.zeros(num_pts)
rho_3p_diff_boson_grid = np.zeros(num_pts)
n_boson_grid = np.zeros(num_pts)
n_der_boson_grid = np.zeros(num_pts)
rho_fermion_grid = np.zeros(num_pts)
rho_der_fermion_grid = np.zeros(num_pts)
p_fermion_grid = np.zeros(num_pts)
rho_3p_diff_fermion_grid = np.zeros(num_pts)
n_fermion_grid = np.zeros(num_pts)
n_der_fermion_grid = np.zeros(num_pts)

@nb.jit(nopython=True, cache=True)
def integrand_rho_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (ET/(math.exp(ET)-1.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_rho_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (ET/(math.exp(ET)+1.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_rho_der_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return ((ET**2.)*math.exp(ET)/((math.exp(ET)-1.)**2.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_rho_der_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return ((ET**2.)*math.exp(ET)/((math.exp(ET)+1.)**2.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_p_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (p**2./(3.*ET*(math.exp(ET)-1.)))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_p_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (p**2./(3.*ET*(math.exp(ET)+1.)))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_rho_3p_diff_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (1./(ET*(math.exp(ET)-1.)))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_rho_3p_diff_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (1./(ET*(math.exp(ET)+1.)))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_n_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (1./(math.exp(ET)-1.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_n_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (1./(math.exp(ET)+1.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_n_der_boson(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (ET*math.exp(ET)/((math.exp(ET)-1.)**2.))*(p**2./(2.*pi2))

@nb.jit(nopython=True, cache=True)
def integrand_n_der_fermion(x, p):
    ET = math.sqrt(x**2. + p**2.)
    return (ET*math.exp(ET)/((math.exp(ET)+1.)**2.))*(p**2./(2.*pi2))

for i, x in enumerate(x_grid):
    print(i)
    integr = lambda p: integrand_rho_boson(x, p)
    rho_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_rho_fermion(x, p)
    rho_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_rho_der_boson(x, p)
    rho_der_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_rho_der_fermion(x, p)
    rho_der_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_p_boson(x, p)
    p_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_p_fermion(x, p)
    p_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_rho_3p_diff_boson(x, p)
    rho_3p_diff_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p:integrand_rho_3p_diff_fermion(x, p)
    rho_3p_diff_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_n_boson(x, p)
    n_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_n_fermion(x, p)
    n_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_n_der_boson(x, p)
    n_der_boson_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

    integr = lambda p: integrand_n_der_fermion(x, p)
    n_der_fermion_grid[i] = integrate.quad(integr, 0., p_max, epsabs=0., epsrel=1e-8)[0]

np.savetxt('rho_red_boson.dat', np.column_stack((x_grid, rho_boson_grid)))
np.savetxt('rho_red_fermion.dat', np.column_stack((x_grid, rho_fermion_grid)))
np.savetxt('rho_der_red_boson.dat', np.column_stack((x_grid, rho_der_boson_grid)))
np.savetxt('rho_der_red_fermion.dat', np.column_stack((x_grid, rho_der_fermion_grid)))
np.savetxt('p_red_boson.dat', np.column_stack((x_grid, p_boson_grid)))
np.savetxt('p_red_fermion.dat', np.column_stack((x_grid, p_fermion_grid)))
np.savetxt('rho_3P_diff_red_boson.dat', np.column_stack((x_grid, rho_3p_diff_boson_grid)))
np.savetxt('rho_3P_diff_red_fermion.dat', np.column_stack((x_grid, rho_3p_diff_fermion_grid)))
np.savetxt('n_red_boson.dat', np.column_stack((x_grid, n_boson_grid)))
np.savetxt('n_red_fermion.dat', np.column_stack((x_grid, n_fermion_grid)))
np.savetxt('n_der_red_boson.dat', np.column_stack((x_grid, n_der_boson_grid)))
np.savetxt('n_der_red_fermion.dat', np.column_stack((x_grid, n_der_fermion_grid)))

# rho_non_rel_grid = np.exp(-x_grid)*np.power(x_grid, 2.5)*(2.*np.pi)**-1.5
# print(pi2/30., rho_boson_grid[0], np.abs(rho_boson_grid[0]-pi2/30.)/rho_boson_grid[0])
# print(pi2*7./240., rho_fermion_grid[0], np.abs(rho_fermion_grid[0]-pi2*7./240.)/rho_fermion_grid[0])
# plt.loglog(x_grid, rho_boson_grid, linestyle="-")
# plt.loglog(x_grid, rho_fermion_grid, linestyle="--")
# plt.loglog(x_grid, rho_non_rel_grid, linestyle='-.')
# plt.show()
# plt.clf()
#
# rho_der_non_rel_grid = np.exp(-x_grid)*(np.power(x_grid, 3.5)+1.5*np.power(x_grid, 2.5))*(2.*np.pi)**-1.5
# print(pi2*2./15., rho_der_boson_grid[0], np.abs(rho_der_boson_grid[0]-pi2*2./15.)/rho_der_boson_grid[0])
# print(pi2*7./60., rho_der_fermion_grid[0], np.abs(rho_der_fermion_grid[0]-pi2*7./60.)/rho_der_fermion_grid[0])
# plt.loglog(x_grid, rho_der_boson_grid, linestyle="-")
# plt.loglog(x_grid, rho_der_fermion_grid, linestyle="--")
# plt.loglog(x_grid, rho_der_non_rel_grid, linestyle='-.')
# plt.show()
# plt.clf()
#
# p_non_rel_grid = np.exp(-x_grid)*np.power(x_grid, 1.5)*(2.*np.pi)**-1.5
# print(pi2/90., p_boson_grid[0], np.abs(p_boson_grid[0]-pi2/90.)/p_boson_grid[0])
# print(pi2*7./(90.*8.), p_fermion_grid[0], np.abs(p_fermion_grid[0]-pi2*7./(90.*8.))/p_fermion_grid[0])
# plt.loglog(x_grid, p_boson_grid, linestyle="-")
# plt.loglog(x_grid, p_fermion_grid, linestyle="--")
# plt.loglog(x_grid, p_non_rel_grid, linestyle='-.')
# plt.show()
# plt.clf()
#
# plt.loglog(x_grid, rho_fermion_grid/(x_grid**4.), linestyle="-")
# plt.loglog(x_grid, pi2*7./(240.*x_grid**4.), linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*np.power(2.*np.pi*x_grid, -1.5), linestyle="-.")
# plt.show()
# plt.clf()
# print(1./12., rho_3p_diff_boson_grid[0])
# print(1./24., rho_3p_diff_fermion_grid[0])
# plt.loglog(x_grid, rho_3p_diff_boson_grid, linestyle="-")
# plt.loglog(x_grid, rho_3p_diff_fermion_grid, linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*((x_grid**0.5)/((2.*np.pi)**1.5) + (3./8.)*(x_grid**(-0.5))/((2.*np.pi)**1.5)), linestyle="-.")
# plt.show()
# plt.clf()
#
# plt.loglog(x_grid, n_boson_grid, linestyle="-")
# plt.loglog(x_grid, np.ones(num_pts)*zeta3/pi2, linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*np.power(x_grid/(2.*np.pi), 1.5), linestyle="-.")
# plt.show()
# plt.clf()
#
# plt.loglog(x_grid, n_fermion_grid, linestyle="-")
# plt.loglog(x_grid, np.ones(num_pts)*0.75*zeta3/pi2, linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*np.power(x_grid/(2.*np.pi), 1.5), linestyle="-.")
# plt.show()
# plt.clf()
#
# plt.loglog(x_grid, n_der_boson_grid, linestyle="-")
# plt.loglog(x_grid, np.ones(num_pts)*3.*zeta3/pi2, linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*(np.power(x_grid, 2.5)+1.5*np.power(x_grid, 1.5))/((2.*np.pi)**1.5), linestyle="-.")
# plt.show()
# plt.clf()
#
# plt.loglog(x_grid, n_der_fermion_grid, linestyle="-")
# plt.loglog(x_grid, np.ones(num_pts)*3.*0.75*zeta3/pi2, linestyle="--")
# plt.loglog(x_grid, np.exp(-x_grid)*(np.power(x_grid, 2.5)+1.5*np.power(x_grid, 1.5))/((2.*np.pi)**1.5), linestyle="-.")
# plt.show()
# plt.clf()
