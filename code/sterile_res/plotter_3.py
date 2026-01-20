#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (FixedLocator, NullLocator, FixedFormatter)
from scipy.interpolate import interp1d

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import constants_functions as cf

xtMajor = np.linspace(0, 3, 4)
xtMinor = [np.log10(i*10**j) for i in range(10)[1:] for j in xtMajor ]
xlMajor = [r"$10^{" + str(int(i)) + "}$" if i in xtMajor else "" for i in xtMajor]
xMajorLocator = FixedLocator(xtMajor)
xMinorLocator = FixedLocator(xtMinor)
xMajorFormatter = FixedFormatter(xlMajor)

ytMajor = np.linspace(-20, -8, 13)
ytMinor = [np.log10(i*10**j) for i in range(10)[1:] for j in ytMajor ]
ylMajor = [r"$10^{" + str(int(i)) + "}$" if i in ytMajor else "" for i in ytMajor]
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

def get_figsize(columnwidth, wf=1.0, hf=(5.**0.5-1.0)/2.0):
    """Parameters:
    - wf [float]:  width fraction in columnwidth units
    - hf [float]:  height fraction in columnwidth units.
                    Set by default to golden ratio.
    - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                            using \showthe\columnwidth
    Returns:  [fig_width, fig_height]: that should be given to matplotlib
    """
    fig_width_pt = columnwidth*wf 
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]

ggplot_red = "#E24A33"
ch = 'crimson' # crimson
c1 = '#797ef6' # orchid
c2 = '#1aa7ec' # sky blue
c3 = '#4adede' # turquoise
c4 = '#ffa62b' # gold
c5 = '#1e2f97' # dark blue

columnwidth = 418.25368     # pt, given by \showthe\textwidth in LaTeX

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
params = {'axes.labelsize': 10,
          'xtick.labelsize': 10,
          'ytick.labelsize': 10,
          'axes.titlesize': 10,
          'font.size': 10 } # extend as needed
# print(plt.rcParams.keys())
plt.rcParams.update(params)


# fig = plt.figure(figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white")
fig = plt.figure(figsize=get_figsize(columnwidth=columnwidth, wf=1.0, hf=0.9), dpi=150, edgecolor="white")
# fig = plt.figure(figsize=get_figsize(columnwidth=columnwidth, wf=1.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis='both', which='both', direction="in", width=0.5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

# nx, ny = 23, 101
# data = np.loadtxt('rm_3.00e+00_y_relic.dat')
# # nx, ny = 23, 111
# # data = np.loadtxt('rm_5.00e+00_y_relic.dat')
# md = data[:,0].reshape((nx, ny))
# mphi = data[:,1].reshape((nx, ny))
# sin22th = data[:,2].reshape((nx,ny))
# y = data[:,3].reshape((nx, ny))
# Odh2 = data[:,4].reshape((nx, ny))
# Odh2_no_spin_stat = data[:,5].reshape((nx, ny))
# xtherm = data[:,6].reshape((nx, ny))
# xdtherm = data[:,7].reshape((nx, ny))
# fs_length = data[:,9].reshape((nx, ny))
# fs_length_3 = data[:,10].reshape((nx, ny))
# T_kd = data[:,11].reshape((nx, ny))
# T_kd_3 = data[:,12].reshape((nx, ny))
# T_d_kd = data[:,13].reshape((nx, ny))
# T_d_kd_3 = data[:,14].reshape((nx, ny))
# r_sound = data[:,15].reshape((nx, ny))
# r_sound_3 = data[:,16].reshape((nx, ny))


nx, ny = 20, 20
data = np.loadtxt('rm_3.00e+00_y_relic_20x20x50.dat')
nx, ny = 40, 40
data = np.loadtxt('rm_3.00e+00_y_relic_40x40x50.dat')
nx, ny = 21, 81
data = np.loadtxt('rm_3.00e+00_y_relic_21x81x60.dat')
nx, ny = 30, 40
data = np.loadtxt('rm_3.00e+00_y_relic_30x40x70_part2.dat')
nx, ny = 20, 30
data = np.loadtxt('rm_3.00e+00_y_relic_20x30x70.dat')
nx, ny = 20, 20
data = np.loadtxt('rm_3.00e+00_y_relic_20x20x50_new.dat')
nx, ny = 20, 40
data = np.loadtxt('rm_3.00e+00_y_relic_20x40x60_new.dat')
nx, ny = 20, 20
data = np.loadtxt('rm_3.00e+00_y_relic_20x20x60_new.dat')
nx, ny = 20, 80
data = np.loadtxt('rm_3.00e+00_y_relic_20x80x70_new.dat')
nx, ny = 20, 20
data = np.loadtxt('rm_3.00e+00_y_relic_20x20x50_new.dat')
nx, ny = 10, 10
data = np.loadtxt('rmXd_5.00e+00_rmhd_3.00e+00_y_relic_test_10x10x70.dat')
nx, ny = 20, 20
data = np.loadtxt('rmXd_5.00e+00_rmhd_3.00e+00_y_relic_test_20x20x60.dat')
nx, ny = 20, 40
data = np.loadtxt('rmXd_5.00e+00_rmhd_3.00e+00_y_relic_test_20x40x60.dat')
nx, ny = 20, 40
data = np.loadtxt('rmXd_3.00e+00_rmhd_3.00e+00_y_relic_test_20x40x60.dat')
nx, ny = 20, 40
data = np.loadtxt('rmXd_3.00e+00_rmhd_3.00e+00_y_relic_test_20x40x60.dat')
nx, ny = 20, 40
data = np.loadtxt('rm_3.00e+00_y_relic_20x40x60_new.dat')

# Removed max_step=1. in pandemolator for this one -- terrible result...
# nx, ny = 30, 30
# data = np.loadtxt('rm_3.00e+00_y_relic_30x30x60_new.dat')

# md = data[:,0].reshape((nx, ny))
# mX = data[:,1].reshape((nx, ny))
# mh = data[:,2].reshape((nx, ny))
# sin22th = data[:,3].reshape((nx,ny))
# y = data[:,4].reshape((nx, ny))
# Odh2 = data[:,5].reshape((nx, ny))
# # Odh2_no_spin_stat = data[:,5].reshape((nx, ny))
# xtherm = data[:,6].reshape((nx, ny))
# xdtherm = data[:,7].reshape((nx, ny))
# fs_length = data[:,9].reshape((nx, ny))
# fs_length_3 = data[:,10].reshape((nx, ny))
# T_kd = data[:,11].reshape((nx, ny))
# T_kd_3 = data[:,12].reshape((nx, ny))
# T_d_kd = data[:,13].reshape((nx, ny))
# T_d_kd_3 = data[:,14].reshape((nx, ny))
# r_sound = data[:,15].reshape((nx, ny))
# r_sound_3 = data[:,16].reshape((nx, ny))

md = data[:,0].reshape((nx, ny))
mX = data[:,1].reshape((nx, ny))
sin22th = data[:,2].reshape((nx,ny))
y = data[:,3].reshape((nx, ny))
Odh2 = data[:,4].reshape((nx, ny))
# Odh2_no_spin_stat = data[:,5].reshape((nx, ny))
xtherm = data[:,5].reshape((nx, ny))
xdtherm = data[:,6].reshape((nx, ny))
fs_length = data[:,8].reshape((nx, ny))
fs_length_3 = data[:,9].reshape((nx, ny))
T_kd = data[:,10].reshape((nx, ny))
T_kd_3 = data[:,11].reshape((nx, ny))
T_d_kd = data[:,12].reshape((nx, ny))
T_d_kd_3 = data[:,13].reshape((nx, ny))
r_sound = data[:,14].reshape((nx, ny))
r_sound_3 = data[:,15].reshape((nx, ny))

# These just became nan for some reason
# r_sound[np.isnan(r_sound)] = 0.34

t_life = 3e12*(1e-10/sin22th)*((1e-6/md)**5.)

# nans, x = np.isnan(y), lambda z: z.nonzero()[0]
# y[nans]= np.interp(x(nans), x(~nans), y[~nans])

from scipy.interpolate import griddata
# Anton: Function for interpolating nan-values in 2D array
def fill_nan(data, method='linear'):
    # Anton: Create x and y coordinate grids for the data
    ny, nx = data.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    
    # Anton: Flatten the arrays for use with griddata
    x_flat = x.flatten()
    y_flat = y.flatten()
    data_flat = data.flatten()
    
    # Anton: Identify valid (non-NaN) points
    valid = ~np.isnan(data_flat)
    points_valid = np.vstack((x_flat[valid], y_flat[valid])).T
    values_valid = data_flat[valid]
    
    # Anton: Points where data is NaN
    points_missing = np.vstack((x_flat[~valid], y_flat[~valid])).T
    
    # Anton: Interpolate the missing data
    data_flat[~valid] = griddata(points_valid, values_valid, points_missing, method=method)
    
    # Anton: Reshape back to the original data shape
    return data_flat.reshape(data.shape)

# y = np.sqrt(2)*md/mX*y

def interpolate_2d(x, y, z, nx=100, ny=100, method='cubic'):
    points = np.array( (x.flatten(), y.flatten()) ).T
    values = z.flatten()
    log_points = np.column_stack([np.log10(points[:,0]), np.log10(points[:,1])])

    grid_logx = np.linspace(np.log10(x.min()), np.log10(x.max()), nx)
    grid_logy = np.linspace(np.log10(y.min()), np.log10(y.max()), ny)
    grid_lx, grid_ly = np.meshgrid(grid_logx, grid_logy)
    grid_lz = griddata(log_points, values, (grid_lx, grid_ly), method=method)

    grid_x = 10**grid_lx
    grid_y = 10**grid_ly
    grid_z = grid_lz  # values are already in linear space
    return grid_x, grid_y, grid_z

# y = np.sqrt(2)*md/mX*y
y = fill_nan(y, method='cubic')
y = fill_nan(y, method='nearest')
r_sound = fill_nan(r_sound, method='cubic')
r_sound = fill_nan(r_sound, method='nearest')
fs_length = fill_nan(fs_length, method='cubic')
fs_length = fill_nan(fs_length, method='nearest')

X, Y, y = interpolate_2d(md, sin22th, y, method='cubic')
_, _, y = interpolate_2d(X, Y, y, method='nearest')
X, Y, r_sound = interpolate_2d(md, sin22th, r_sound, method='cubic')
_, _, r_sound = interpolate_2d(X, Y, r_sound, method='nearest')
X, Y, fs_length = interpolate_2d(md, sin22th, fs_length, method='cubic')
md, sin22th, fs_length = interpolate_2d(X, Y, fs_length, method='cubic')

y *= 0.9
mX = 5*md
mh = 3*md

plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(y), levels=[-6.,-5.,-4.,-3.,-2.,-1.], colors='forestgreen', linewidths = 0.4, zorder=-1, linestyles='-')

# plt.contour(np.log10(1e6*md), np.log10(sin22th), np.abs(Odh2_no_spin_stat-Odh2)/Odh2, levels=[0.1])
# plt.contour(np.log10(1e6*md), np.log10(sin22th), t_life, levels=[1.])

# LYMAN-ALPHA
plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(fs_length), levels=[np.log10(0.24)], colors='darkorange', linewidths=1.3, zorder=-3, linestyles='-')
plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(r_sound), levels=[np.log10(0.34)], colors='#ff7b7b', linewidths=1.3, zorder=-4, linestyles='-')

plt.contourf(np.log10(1e6*md), np.log10(sin22th), np.log10(fs_length), levels=[np.log10(0.24), 6.], colors='darkorange', alpha=0.25, zorder=-3)
plt.contourf(np.log10(1e6*md), np.log10(sin22th), np.log10(r_sound), levels=[np.log10(0.34), 6.], colors='#ff7b7b', alpha=0.25, zorder=-4)

# DODELSON-WIDROW
dw_mid = np.loadtxt('../data/dw/0612182_dw_fig_5.dat', skiprows=2)
dw_mid[:,1] = (0.12/0.11)*dw_mid[:,1]*((1e-6/dw_mid[:,0])**2.)
dw_up = np.loadtxt('../data/dw/0612182_dw_fig_5_up.dat', skiprows=2)
dw_up[:,1] = (0.12/0.105)*dw_up[:,1]*((1e-6/dw_up[:,0])**2.)
dw_low = np.loadtxt('../data/dw/0612182_dw_fig_5_low.dat', skiprows=2)
dw_low[:,1] = (0.12/0.105)*dw_low[:,1]*((1e-6/dw_low[:,0])**2.)
dw_region = np.concatenate((dw_low, dw_up[::-1]))
# -->
plt.plot(np.log10(1e6*dw_mid[:,0]), np.log10(dw_mid[:,1]), color='#83781B', ls='--', zorder=1)
plt.plot(np.log10(1e6*dw_low[:,0]), np.log10(dw_low[:,1]), color='#83781B', ls=':', zorder=1)
plt.plot(np.log10(1e6*dw_up[:,0]), np.log10(dw_up[:,1]), color='#83781B', ls=':', zorder=1)
plt.fill_between(np.log10(1e6*dw_region[:,0]), np.log10(dw_region[:,1]), color='#EAE299', ls=':', zorder=0)

# OMEGA_S H^2s
skyblue_alpha08 = '#92CBE2'     # (color=skyblue, alpha=0.8)
plt.fill_between(np.log10(1e6*dw_low[:,0]), np.log10(dw_low[:,1]), color=skyblue_alpha08, edgecolor='none', zorder=-1, lw=0, alpha=1)
plt.fill_between(np.log10(1e6*dw_up[:,0]), np.log10(dw_up[:,1]), color='#92CBE2', edgecolor='none', zorder=-1, lw=0, alpha=1)

# X-RAY CONSTRAINTS
constraint = np.loadtxt('../../xray_constraints/overall_constraint.dat')
plt.fill_between(np.log10(1e6*constraint[:,0]), np.log10(constraint[:,1]), 1e0, color='white', lw=1.3, alpha=1, zorder=-2)
plt.fill_between(np.log10(1e6*constraint[:,0]), np.log10(constraint[:,1]), 1e0, color='black', lw=1.3, alpha=0.25, zorder=-2)
plt.plot(np.log10(1e6*constraint[:,0]), np.log10(constraint[:,1]), color='black', lw=1.3, zorder=-2)

# X-RAY PROJECTIONS
athena_proj = np.loadtxt('../../xray_constraints/Athena_projection_2103.13242.dat', skiprows=2)
erosita_proj = np.loadtxt('../../xray_constraints/eROSITA_projection_2103.13241.dat', skiprows=2)
extp_proj = np.loadtxt('../../xray_constraints/eXTP_projection_2001.07014.dat', skiprows=2)
plt.plot(np.log10(1e6*athena_proj[:,0]), np.log10(athena_proj[:,1]), color='black', lw=1.3, ls='-.', zorder=1)
plt.plot(np.log10(1e6*erosita_proj[:,0]), np.log10(erosita_proj[:,1]), color='black', lw=1.3, ls='--', zorder=1)
plt.plot(np.log10(1e6*extp_proj[:,0]), np.log10(extp_proj[:,1]), color='black', lw=1.3, ls=':', zorder=1)

# SELF-INTERACTIONS
self_int_const = cf.conv_cm2_g
# sigma_self_int = (y**4.)*(np.cos(0.5*np.arcsin(np.sqrt(sin22th)))**8.)*md/(4.*np.pi*(mX**4.))
# sigma_self_int = 3*sigma_self_int       # Vector mediator give 3x enhancement
sigma_self_int = (4*y**4*np.cos(0.5*np.arcsin(np.sqrt(sin22th)))**8.) * md*(mh**2-md**2)**2/(np.pi*mh**4*mX**4)
plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(sigma_self_int), levels=[np.log10(self_int_const)], colors='#A300CC', linewidths=1.3, zorder=-5)
plt.contourf(np.log10(1e6*md), np.log10(sin22th), np.log10(sigma_self_int), levels=[np.log10(self_int_const), np.log10(1e6*self_int_const)], colors='#A300CC', alpha=0.25, zorder=-5)

def get_xy(cs):
    p = cs.get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]

    return x, y

cs1 = plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(fs_length), levels=[np.log10(0.12)], colors='none', linewidths=1.3, zorder=1, linestyles='--')
x1, y1 = get_xy(cs1)

cs2 = plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(r_sound), levels=[np.log10(0.15)], colors='none', linewidths=1.3, zorder=1, linestyles='--')
x2, y2 = get_xy(cs2)

cs3 = plt.contour(np.log10(1e6*md), np.log10(sin22th), np.log10(sigma_self_int), levels=[np.log10(0.1*self_int_const)], colors='none', linewidths=1.3, linestyles='--')
x3, y3 = get_xy(cs3)

# plt.plot(x1[:-17], y1[:-17])
# plt.plot(x2[71:-18], y2[71:-18])
# plt.plot(x3[18:], y3[18:])

ip1 = interp1d(y1, x1, kind='linear')
ip2 = interp1d(y2, x2, kind='linear')
ip3 = interp1d(y3, x3, kind='linear')

Y1 = np.linspace(-14.38, -12, 100)
X1 = [ip1(y) for y in Y1]
Y2 = np.linspace(-16.05, -14.38, 100)
X2 = [ip2(y) for y in Y2]
Y3 = np.linspace(-17.16, -16.05, 100)
X3 = [ip3(y) for y in Y3]

plt.plot(X1, Y1, zorder=-3, color='darkorange', linestyle='--')
plt.plot(X2, Y2, zorder=-3, color='#ff7b7b', linestyle='--')
plt.plot(X3, Y3, zorder=-3, color='#A300CC', linestyle='--')

load_str_1 = './md_1.12884e-05;mX_5.64419e-05;mh_3.38651e-05;sin22th_1.83298e-13;y_1.93457e-04;full_new.dat' 
load_str_2 = './md_2.1e-05;mX_1.05e-04;mh_6.3e-05;sin22th_1.5e-15;y_1.282e-03;full_new.dat'       
load_str_3 = './md_4e-06;mX_2e-05;mh_1.2e-05;sin22th_3e-15;y_8.36e-04;full_new.dat'    
load_str_4 = './md_5e-05;mX_1.5e-04;mh_9e-05;sin22th_5e-16;y_1.24e-03;full_new.dat'    
load_str_4 = './md_5.13483e-05;mX_2.56742e-04;mh_1.54045e-04;sin22th_3.66524e-16;y_3.36087e-03;full_new.dat' 
load_str_4 = './md_4e-05;mX_2.56742e-04;mh_1.54045e-04;sin22th_3e-12;y_3.36087e-03;full_new.dat' 

# load_str_3 = './md_1e-05;mX_5e-04;mh_3e-04;sin22th_5e-16;y_3.5e-03;full_new.dat'       

# BENCHMARK POINTS
md_B1, sin22th_B1 = load_str_1.replace('./', '').split(';')[0], load_str_1.replace('./', '').split(';')[3]
md_B2, sin22th_B2 = load_str_2.replace('./', '').split(';')[0], load_str_2.replace('./', '').split(';')[3]
md_B3, sin22th_B3 = load_str_3.replace('./', '').split(';')[0], load_str_3.replace('./', '').split(';')[3]
md_B4, sin22th_B4 = load_str_4.replace('./', '').split(';')[0], load_str_4.replace('./', '').split(';')[3]

md_B1, sin22th_B1 = float(md_B1.replace('md_', '')), float(sin22th_B1.replace('sin22th_', ''))
md_B2, sin22th_B2 = float(md_B2.replace('md_', '')), float(sin22th_B2.replace('sin22th_', ''))
md_B3, sin22th_B3 = float(md_B3.replace('md_', '')), float(sin22th_B3.replace('sin22th_', ''))
md_B4, sin22th_B4 = float(md_B4.replace('md_', '')), float(sin22th_B4.replace('sin22th_', ''))

plt.plot(np.log10(1e6*md_B1), np.log10(sin22th_B1), marker='*', color='tomato')
plt.plot(np.log10(1e6*md_B2), np.log10(sin22th_B2), marker='*', color='tomato')
plt.plot(np.log10(1e6*md_B3), np.log10(sin22th_B3), marker='*', color='tomato')
# plt.plot(np.log10(1e6*md_B4), np.log10(sin22th_B4), marker='*', color='tomato')

#plt.plot(np.log10(10), np.log10(3.5e-13), marker='*', color='tomato')
#plt.plot(np.log10(20), np.log10(3.5e-15), marker='*', color='tomato')

ax.text(0.08, -9.4, r"$g_X = 10^{-6}$", color='forestgreen', zorder=-1)
ax.text(0.201, -11.0, r"$10^{-5}$", color='forestgreen', zorder=-1)
ax.text(0.201, -12.83, r"$10^{-4}$", color='forestgreen', zorder=-1)
ax.text(0.201, -14.72, r"$10^{-3}$", color='forestgreen', zorder=-1)
ax.text(0.201, -17.30, r"$10^{-2}$", color='forestgreen', zorder=-1)
ax.text(2, -17.80, r"$10^{-1}$", color='forestgreen', zorder=-1)

ax.text(0.53, -13.4, r"Ly-$\alpha$", color='darkorange')
ax.text(0.53, -13.8, r"$\lambda_\mathrm{fs}$", color='darkorange')
# ax.text(0.06, -13.6, r"Ly-$\alpha$", color='#ff7b7b', zorder=-4)
# ax.text(0.06, -14.0, r"$r_\text{s}$", color='#ff7b7b', zorder=-4)
ax.text(0.35, -15.2, r"Ly-$\alpha$", color='#ff7b7b', zorder=-4)
ax.text(0.35, -15.6, r"$r_\text{s}$", color='#ff7b7b', zorder=-4)
# ax.text(0.4, -16.45, r"self-interactions", color='#A300CC', rotation=-9)
ax.text(0.8, -17.5, r"self-interactions", color='#A300CC', rotation=0)
ax.text(1.5, -10.25, r"Dodelson-Widrow", color='#83781B', rotation=-22, ha='center')
ax.text(1.45, -13., r"X-rays", color='black', rotation=0)
# ax.text(1.6, -10.15, r"$\Omega_s h^2 > 0.12$", color='#155D7A', rotation=-22)
ax.text(1.6, -10.25, r"overproduction", color='#155D7A', rotation=-24)
ax.text(0.3, -10.69, "eROSITA", color='black', rotation=-45)
ax.text(0.95, -13.35, "Athena", color='black', rotation=-15)
ax.text(1.04, -15., "eXTP", color='black', rotation=0)

plt.xlim(0, np.log10(300))
plt.ylim(-18, -8)

props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")
ax.text(2.05, -8.4, "$m_X = 5 m_s$" + "\n" + "$m_h \hspace{1mm}= 3 m_s$", color='black', bbox=props, va='top')
# ax.text(1.9, -7.75, "$m_h = 3 m_s$", color='black', fontsize=12, bbox=props)
ax.xaxis.set_label_text(r"$m_s\;\;[\mathrm{keV}]$")
ax.xaxis.set_major_locator(xMajorLocator)
ax.xaxis.set_minor_locator(xMinorLocator)
ax.xaxis.set_major_formatter(xMajorFormatter)

ax.yaxis.set_label_text(r"$\sin^2 (2 \theta)$")
ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)
plt.tight_layout()
savefig_str = './find_y_md_sin22th_' + f'mX_{(mX/md)[0,0]:.1f}md' + f'mh_{(mh/md)[0,0]:.1f}md.pdf'
print('saved', savefig_str)
# plt.savefig(savefig_str, dpi=300)
plt.show()
