#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

# CONSTRAINTS
cbs = np.loadtxt('CBS_2008.02283.dat', skiprows=2)
dwarfs = np.loadtxt('Dwarfs_1408.3531.dat', skiprows=2)
integral = np.loadtxt('INTEGRAL_0710.4922.dat', skiprows=2)
m31 = np.loadtxt('M31_1311.0282.dat', skiprows=2)
nustar = np.loadtxt('NuSTAR_1908.09037.dat', skiprows=2)
xmmblanksky = np.loadtxt('XMMBlankSky_2102.02207.dat', skiprows=2)

cbs_interp = interp1d(np.log(cbs[:,0]), np.log(cbs[:,1]), kind='linear', bounds_error=False, fill_value=10.)
dwarfs_interp = interp1d(np.log(dwarfs[:,0]), np.log(dwarfs[:,1]), kind='linear', bounds_error=False, fill_value=10.)
m31_interp = interp1d(np.log(m31[:,0]), np.log(m31[:,1]), kind='linear', bounds_error=False, fill_value=10.)
integral_interp = interp1d(np.log(integral[:,0]), np.log(integral[:,1]), kind='linear', bounds_error=False, fill_value=10.)
nustar_interp = interp1d(np.log(nustar[:,0]), np.log(nustar[:,1]), kind='linear', bounds_error=False, fill_value=10.)
xmmblanksky_interp = interp1d(np.log(xmmblanksky[:,0]), np.log(xmmblanksky[:,1]), kind='linear', bounds_error=False, fill_value=10.)

def overall_interp(m_d):
    log_m_d = np.log(m_d)

    cbs_dwarfs = np.minimum(cbs_interp(log_m_d), dwarfs_interp(log_m_d))
    m31_integral = np.minimum(m31_interp(log_m_d), integral_interp(log_m_d))
    integral_nustar = np.minimum(nustar_interp(log_m_d), xmmblanksky_interp(log_m_d))

    overall = np.minimum(np.minimum(cbs_dwarfs, m31_integral), integral_nustar)

    return np.exp(overall)

min_m_d = min(cbs[0,0], dwarfs[0,0], integral[0,0], m31[0,0], nustar[0,0], xmmblanksky[0,0])
max_m_d = max(cbs[-1,0], dwarfs[-1,0], integral[-1,0], m31[-1,0], nustar[-1,0], xmmblanksky[-1,0])
m_d_grid = np.logspace(np.log10(min_m_d), np.log10(max_m_d), 10000)
overall_constraint_grid = overall_interp(m_d_grid)
np.savetxt('overall_constraint.dat', np.column_stack((m_d_grid, overall_constraint_grid)))

# PROJECTIONS
anita_proj = np.loadtxt('Athena_projection_2103.13242.dat', skiprows=2)
erosita_proj = np.loadtxt('eROSITA_projection_2103.13241.dat', skiprows=2)
extp_proj = np.loadtxt('eXTP_projection_2001.07014.dat', skiprows=2)

plt.loglog(cbs[:,0], cbs[:,1], ls='-')
plt.loglog(dwarfs[:,0], dwarfs[:,1], ls='-')
plt.loglog(integral[:,0], integral[:,1], ls='-')
plt.loglog(m31[:,0], m31[:,1], ls='-')
plt.loglog(nustar[:,0], nustar[:,1], ls='-')
plt.loglog(xmmblanksky[:,0], xmmblanksky[:,1], ls='-')

plt.loglog(anita_proj[:,0], anita_proj[:,1], ls='--')
plt.loglog(erosita_proj[:,0], erosita_proj[:,1], ls='--')
plt.loglog(extp_proj[:,0], extp_proj[:,1], ls='--')
plt.loglog([extp_proj[-1,0], 10.*extp_proj[-1,0]], [extp_proj[-1,1], 1e-5*extp_proj[-1,1]], ls='--')

plt.show()
