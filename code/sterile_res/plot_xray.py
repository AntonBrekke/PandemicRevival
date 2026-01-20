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


m_d = 1e-6*np.logspace(0, 2.5, 100)    # 10^a keV - 10^b keV, a=0, b=2.5
sin2_2th = np.logspace(-18, -8, 100)

m_d, sin2_2th = np.meshgrid(m_d, sin2_2th)
plt.plot(m_d, sin2_2th, np.ones_like(m_d)*np.nan, color='black', lw=0.5, zorder=0)

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

plt.xlim(0, np.log10(300))
plt.ylim(-18, -8)

ax.text(1.5, -10.25, r"Dodelson-Widrow", color='#83781B', rotation=-22, ha='center')
ax.text(1.45, -13., r"X-rays", color='black', rotation=0)
ax.text(1.6, -10.25, r"overproduction", color='#155D7A', rotation=-24)

ax.xaxis.set_label_text(r"$m_s\;\;[\mathrm{keV}]$")
ax.xaxis.set_major_locator(xMajorLocator)
ax.xaxis.set_minor_locator(xMinorLocator)
ax.xaxis.set_major_formatter(xMajorFormatter)

ax.yaxis.set_label_text(r"$\sin^2 (2 \theta)$")
ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)
plt.tight_layout()

plt.show()