#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, asin, sqrt, exp
from scipy.special import kn
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 8.0

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

data1 = np.loadtxt('rates_md_1.2000e-05_mphi_3.6000e-05_sin22th_2.5000e-13_y_1.9050e-04_2to4.dat')
x_grid1 = data1[:,0]
H1 = data1[:,1]
C_p_dd1 = data1[:,2]
C_dd_p1 = data1[:,3]
C_p_da1 = data1[:,4]
C_da_p1 = data1[:,5]
C_p_aa1 = data1[:,6]
C_aa_p1 = data1[:,7]
C_pp_dd1 = data1[:,8]
C_dd_pp1 = data1[:,9]
C_dd_dd1 = data1[:,10]
C_da_dd1 = data1[:,11]
C_dd_da1 = data1[:,12]
C_aa_dd1 = data1[:,13]
C_dd_aa1 = data1[:,14]
C_2p_4p1 = data1[:,15]

data2 = np.loadtxt('rates_md_2.0000e-05_mphi_6.0000e-05_sin22th_3.0000e-15_y_1.6022e-03_2to4.dat')
x_grid2 = data2[:,0]
H2 = data2[:,1]
C_p_dd2 = data2[:,2]
C_dd_p2 = data2[:,3]
C_p_da2 = data2[:,4]
C_da_p2 = data2[:,5]
C_p_aa2 = data2[:,6]
C_aa_p2 = data2[:,7]
C_pp_dd2 = data2[:,8]
C_dd_pp2 = data2[:,9]
C_dd_dd2 = data2[:,10]
C_da_dd2 = data2[:,11]
C_dd_da2 = data2[:,12]
C_aa_dd2 = data2[:,13]
C_dd_aa2 = data2[:,14]
C_2p_4p2 = data2[:,15]

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=14)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fig = plt.figure(figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1,1,1)
ax.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)

ytMajor = np.array([np.log10(10**j) for j in np.linspace(-28, -5, 24)])
ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
ylMajor = [r"$10^{" + str(int(i)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
ytMajor = 10**ytMajor
ytMinor = 10**ytMinor
yMajorLocator = FixedLocator(ytMajor)
yMinorLocator = FixedLocator(ytMinor)
yMajorFormatter = FixedFormatter(ylMajor)

"""
ch = 'darkgreen' #'mediumseagreen'
c1 = '#80b1d3' #'#5170d7'
c3 = '#fb8072' #'crimson'
c2 = '#bc80bc' #'mediumorchid'
c5 = '#fdb362' #'#f0944d'
"""
"""
ch = '#f6593f'
c1 = '#142772' #'#5170d7'
c3 = '#2161a9' #'crimson'
c2 = '#37acc2' #'mediumorchid'
c5 = '#7acbbb' #'#f0944d'
"""

ch = 'crimson'
c1 = '#797ef6' #'#5170d7'
c3 = '#4adede' #'crimson'
c2 = '#1aa7ec' #'mediumorchid'
c5 = '#1e2f97' #'#f0944d'
c6 = '#458751'

c4 = '#ffa62b'

plt.loglog(x_grid1, 1e6*H1, color=ch, ls='-', zorder=0) #83781B
plt.loglog(x_grid1, 1e6*C_dd_p1, color=c1, ls='-', zorder=-4) #114B5F
plt.loglog(x_grid1, 1e6*C_da_p1, color=c2, ls='-', zorder=-4) #458751
# plt.loglog(x_grid1, 1e6*C_p_da1, color='#53A262', ls='-')
plt.loglog(x_grid1, 1e6*C_dd_pp1, color=c3, ls='-', zorder=-4) #95190C
plt.loglog(x_grid1, 1e6*C_pp_dd1, color=c4, ls='-', zorder=-4) #D02411
plt.loglog(x_grid1, 1e6*C_2p_4p1, color=c6, ls='-', zorder=-4)

plt.loglog(x_grid2, 1e6*H2, color=ch, ls='--', zorder=0)
plt.loglog(x_grid2, 1e6*C_dd_p2, color=c1, ls='--', zorder=-4)
plt.loglog(x_grid2, 1e6*C_da_p2, color=c2, ls='--', zorder=-4)
# plt.loglog(x_grid2, 1e6*C_p_da2, color='#53A262', ls='--')
plt.loglog(x_grid2, 1e6*C_dd_pp2, color=c3, ls='--', zorder=-4)
plt.loglog(x_grid2, 1e6*C_pp_dd2, color=c5, ls='--', zorder=-4)
plt.loglog(x_grid2, 1e6*C_2p_4p2, color=c6, ls='--', zorder=-4)

plt.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
plt.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
#plt.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


ax.text(2e-4, 3e-14, r"$H$", color=ch, fontsize=10, rotation=0)
ax.text(9e-2, 2e-13, r"$\nu_s \nu_s \leftrightarrow \phi$", color=c1, fontsize=10, rotation=0)
ax.text(1.5e-3, 3e-25, r"$\nu_s \nu_\alpha \to \phi$", color=c2, fontsize=10, rotation=0)
ax.text(1.5e-3, 0.3e-19, r"$\nu_s \nu_s \to \phi \phi$", color=c3, fontsize=10, rotation=0)
ax.text(2.2e-3, 2e-28, r"$\phi \phi \to \nu_s \nu_s$", color=c5, fontsize=10, rotation=0)
ax.text(1.5e-3, 2e-26, r"$2 \phi \to 4 \phi$", color=c6, fontsize=10, rotation=0)

plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-', color='black', label=r'$\textit{BP1}$')
plt.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\textit{BP2}$')


plt.fill_betweenx([1e-28, 1e-8], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
plt.loglog([1e-3]*2, [1e-28, 1e-8], ls=':', color='0', zorder=-2)
#plt.fill_betweenx([1e-28, 1e-8], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

# N = 1000
# for i in range(N)[:-1]:
#     x1 = -7 + (-3.5 - (-7))*i/(N-1)
#     x2 = -7 + (-3.5 - (-7))*(i+1)/(N-1)
#
#     plt.fill_betweenx([1e-28, 1e-6], 10**x1, 10**x2, color='white', alpha=1-i/N, zorder=1)
# plt.fill_betweenx([1e-28, 1e-6], 1e-5, 3e-5, color='white', alpha=1, zorder=1)


props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")
# ax.text(6e-1, 9e-18, r"$m_s = 15 \, \mathrm{keV}$" + "\n" + r"$m_\phi = 37.5 \, \mathrm{keV}$" + "\n" + r"$\sin^2 (2 \theta) = 1.5 \times 10^{-13}$" + "\n" + r"$y = 2.44 \times 10^{-4}$", horizontalalignment="left", fontsize=10, bbox=props)

plt.legend(fontsize=9, framealpha=0.8, edgecolor='none')

ax.xaxis.set_label_text(r"$m_s / T_\nu$")
ax.yaxis.set_label_text(r"$\text{Rate} \;\; [\mathrm{keV}]$")

ax.yaxis.set_major_locator(yMajorLocator)
ax.yaxis.set_minor_locator(yMinorLocator)
ax.yaxis.set_major_formatter(yMajorFormatter)

plt.xlim(2e-5, 20)
ax.set_ylim(1e-28, 1e-8)
plt.tight_layout()
plt.savefig('rates_2to4.pdf')
plt.show()
