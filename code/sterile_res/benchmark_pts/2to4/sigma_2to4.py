#! /usr/bin/python3

# math
from math import log10, pow, exp, log
# numpy
import numpy as np
# scipy
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.special import kn
# numba
import numba as nb
# filesystem
import os
import inspect

# params
approx_zero = 1e-200; approx_zero_log = np.log10(approx_zero)


# Define the reference mass
m0 = 100

# 2TO4 (SYMMETRIC) CROSS-SECTION ##############################################

# 0. SPECIFY THE DATA
_N = 3000
# Each range has the same number of points
_Epts_log = np.array([
    2.301030039093427071e+00,
    2.301854372269586779e+00,
    2.863917376957860306e+00,
    6.000000000000000000e+00
])

# -->
_Npts = len(_Epts_log) - 1

# 1. READ THE DATA
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_data = np.loadtxt(currentdir+'/2to4_100MeV.dat')

_E_log, _s_log = _data[:,0], _data[:,1]
# Calculate \sigma * E^2
_S_log = _s_log + 2*_E_log


# 2. DEFINE THE INTERPOLATION ROUTINE
@nb.njit(cache=True)
def _get_index(E_log):
    # First, we have to find the correct position in _Epts
    # Here, we can assume that E_log is contained in the data
    # This is ensured by the function _s0_interp below
    pos = 0
    for i in range(_Npts):
        if _Epts_log[i] <= E_log <= _Epts_log[i+1]:
            pos = i
            break

    # From this we now determine the offset in the data
    # pos = 0 starts at 0
    # pos = 1 starts at _Npts-1 (overlaps with the previous range)
    # -->
    # pos = i starts at (_N-1)*pos
    offset = (_N-1)*pos

    # Now lets get the index
    Emin_log = _Epts_log[pos  ]
    Emax_log = _Epts_log[pos+1]
    # -->
    index = int( ( _N - 1 ) * ( E_log - Emin_log ) / ( Emax_log - Emin_log ) )
    # Handle the boundary:
    # Since there always needs to be a point at index+1
    # we have to substract 1 if we land on the final index
    index = index if index != _N - 1 else index - 1

    return index + offset


@nb.njit
def _s0_interp(E):
    E_log = log10(E)

    if E_log < _Epts_log[0]:
        return pow(10., approx_zero_log - 2.*E_log)

    if E_log > _Epts_log[-1]:
        return pow(10., _S_log[-1] - 2.*E_log)

    i = _get_index(E_log)

    # Extract the relevant values
    x1, x2 = _E_log[i], _E_log[i+1]
    y1, y2 = _S_log[i], _S_log[i+1]

    # Perform the interpolation
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1

    # We interpolate log10(sigma*(E^2)) vs log10(E);
    # hence, we have to divide the potentiated result
    # by E^2, which leads to the -2 in the slope
    return pow(10., (m-2)*E_log + b)

# 3. DEFINE THE FINAL CROSS-SECTION
@nb.njit
def s2to4(l, m, E):
    return (l**4) * _s0_interp( E*m0/m ) * (m0/m)**2.

# m: mass of phi in GeV
# T: temperature in GeV
# returns collision operator in GeV^4
def C2to4(l, m, T, xi):
    m = 1e3*m
    T = 1e3*T
    log_E_min = log(2.*m)
    log_E_max = log(3e2*T)
    if log_E_max <= log_E_min:
        return 0.
    
    def integrand(log_E):
        E = exp(log_E)
        sigma = s2to4(l, m, E)

        return sigma*(E**3.)*(E*E-m*m)*kn(1, 2.*E/T)
    
    sigma_v_part, err = quad(integrand, log_E_min, log_E_max, epsabs=0., epsrel=1e-4, limit=100, points=log_E_min)

    return 1e-12*2.*T*exp(2.*xi)*sigma_v_part/(np.pi**4.)

if __name__ == '__main__':
    print(s2to4(1, 50, 200))