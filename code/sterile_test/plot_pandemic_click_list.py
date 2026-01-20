import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
import tkinter as tkint
from pathlib import Path

"""
Mostly a tool to quickly look at plots
For better quality plots (sharper, no bugs) run plot_pandemic.py
"""

import matplotlib
matplotlib.rcParams['hatch.linewidth'] = 8.0
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
# # plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

import os
import sys
import inspect

# To import constant_functions 
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, grandparentdir)

import constants_functions as cf

def plot_pandemic(load_str):

    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(0.4*12.0, 0.4*11.0), dpi=150, edgecolor="white", gridspec_kw={'height_ratios': [2, 1]})
    ax1.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
    ax1.yaxis.set_ticks_position('both')
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(0.5)
    ax2.tick_params(axis='both', which='both', labelsize=11, direction="in", width=0.5)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('both')
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(0.5)

    xtMajor = np.array([np.log10(10**j) for j in np.linspace(-5, 2, 8)])
    xtMinor = np.array([np.log10(i*10**j) for j in xtMajor for i in range(10)[1:10]])
    xlMajor = [r"$10^{" + str(int(i)) + "}$" if i in xtMajor else "" for i in xtMajor]
    xtMajor = 10**xtMajor
    xtMinor = 10**xtMinor
    xMajorLocator = FixedLocator(xtMajor)
    xMinorLocator = FixedLocator(xtMinor)
    xMajorFormatter = FixedFormatter(xlMajor)

    # Anton: Formatting is to move y-tick scale from GeV to keV (str(int(i+6)))
    ytMajor = np.array([np.log10(10**j) for j in np.linspace(-25, 5, 25-(-5)+1)])
    ytMinor = np.array([np.log10(i*10**j) for j in ytMajor for i in range(10)[1:10]])
    ylMajor = [r"$10^{" + str(int(i+6)) + "}$" if i in ytMajor[::2] else "" for i in ytMajor]
    ytMajor = 10**ytMajor
    ytMinor = 10**ytMinor
    yMajorLocator = FixedLocator(ytMajor)
    yMinorLocator = FixedLocator(ytMinor)
    yMajorFormatter = FixedFormatter(ylMajor)


    data = np.loadtxt(load_str)

    T_SM = data[:,1]
    T_nu = data[:,2]
    ent = data[:,3]
    Td = data[:,6]
    xid = data[:,7]
    xiX = data[:,8]
    nd = data[:,9]
    nX = data[:,10]

    var_list = load_str.split(';')[:-1]
    # md, mX, mh, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
    md, mX, sin22th, y = [eval(s.split('_')[-1]) for s in var_list]
    # print(f'md: {md:.2e}, mX: {mX:.2e}, sin22th: {sin22th:.2e}, y: {y:.2e}')

    T_grid_dw = np.logspace(np.log10(1.4e-3), 1, 400)
    mYd_dw = cf.O_h2_dw_Tevo(T_grid_dw, md, 0.5*np.arcsin(np.sqrt(sin22th)))*cf.rho_crit0_h2 / cf.s0  
    mY_relic = cf.omega_d0 * cf.rho_crit0_h2 / cf.s0        # m*Y = m*n/s = Omega * rho_c0 / s0

    x1_dw = md/T_grid_dw
    y1_dw = mYd_dw

    x1_tr = md/T_nu
    y1_tr = md*nd/ent

    x1_dw0, x1_tr0 = x1_dw[x1_dw < 1e-3], x1_tr[x1_tr > 1e-3]
    y1_dw0, y1_tr0 = y1_dw[x1_dw < 1e-3], y1_tr[x1_tr > 1e-3]

    x1, y1 = np.array([*x1_dw0[::-1], *x1_tr0, 1e3]), np.array([*y1_dw0[::-1], *y1_tr0, y1_tr0[-1]])

    # ax1.loglog(x1, y1, color='#7bc043', zorder=-1)
    ax1.loglog(x1_tr, y1_tr, color='r', zorder=-1)
    ax1.loglog(x1_dw, y1_dw, color='#7bc043', zorder=-1)
    # ax1.loglog(x1_dw[x1_dw < 1e-3], y1_dw[x1_dw < 1e-3], color='r', zorder=-1)

    # ax1.fill_betweenx([1e-23, 1e5], 1e-5, 1e-3, color='white', alpha=1, zorder=-3)
    # ax1.fill_betweenx([1e-23, 1e-18], 1e-5, 1e-3, facecolor="white", hatch="\\", edgecolor="0.9", zorder=1)

    #ax1.text(8e-5, 2e-21, 'Thermalization', fontsize=10, color='darkorange')
    #ax1.text(5e-4, 1.3e-19, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    #ax1.text(5e-4, 1e-22, r'$\rightarrow$', color='darkorange', horizontalalignment='center', verticalalignment='center')
    ax1.axvline([2e-3], ls=':', color='0', zorder=-2)
    ax2.axvline([2e-3], ls=':', color='0', zorder=-2)
    # ax1.plot([1e-3]*2, [1e-25, 2e-8], ls=':', color='0', zorder=-2)
    # ax2.plot([1e-3]*2, [1e-2, np.max(Td/T_nu)*1.5], ls=':', color='0', zorder=-2)

    ax1.text(1.5e-4, 8e-21, r'$\mathrm{Dark}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-22, r'$\mathrm{Thermalization}$', fontsize=8, color='0', horizontalalignment='center')
    ax1.text(1.5e-4, 8e-23, r'$\rightarrow$', fontsize=8, color='0', horizontalalignment='center')
    #ax1.text(4.5e-5, 1e-22, r'$\hspace{-0.55cm}\mathrm{Therma-}\\\mathrm{lization}\\\mathrm{ }\hspace{0.2cm}\rightarrow$', fontsize=10, color='0')


    ax1.loglog(md/T_nu, mX*nX/ent, color='#f37736', ls='-', zorder=-4)

    # 1.1*mY_relic just for the eye, some sort of rendering bug, but data is exactly mY_relic in plot_pandemic.py
    ax1.loglog([1e-8, 1e3], [1.1*mY_relic, 1.1*mY_relic], color='0.65', ls='-.', zorder=-2)
    ax1.text(3e-5, 1e-11, r'$\Omega_s h^2 = 0.12$', fontsize=11, color='0.65')

    ax1.text(2.5, 1e-11, r'$\nu_s$', color='#7bc043', fontsize=11)
    ax1.text(2.0, 1e-16, r'$X$', color='#f37736', fontsize=11)

    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='-' , color='black', label=r'$\text{BP1}$')
    ax2.plot([1e-10, 1e-9], [1e-40, 1e-35], linestyle='--', color='black', label=r'$\text{BP2}$')

    ax2.loglog(md/T_nu, Td/T_nu, color='0.4', ls='-', zorder=-4)

    ax2.fill_betweenx([1e-1, 1.5e0], 1e-5, 2e-3, color='white', alpha=1, zorder=-3)

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=1, edgecolor="0.8")

    #plt.text(2e-2, 1e-11, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props)
    #plt.text(1e0, 3e-23, r'$m_X = 2.5m_\chi$', fontsize=9, horizontalalignment='center', bbox=props, zorder=5)
    #ax1.text(1e0, 8e-25, r'$m_X = 2.5m_\chi$', fontsize=10, horizontalalignment='center', zorder=5)


    ax2.legend(fontsize=10, framealpha=0.8, edgecolor='1')
    ax2.xaxis.set_label_text(r"$m_s / T_\nu$")
    ax1.yaxis.set_label_text(r"$m\, n / s\;\;\mathrm{[keV]}$")
    ax2.yaxis.set_label_text(r"$T_\text{d}/T_\nu$")


    ax1.xaxis.set_major_locator(xMajorLocator)
    ax1.xaxis.set_minor_locator(xMinorLocator)
    ax1.xaxis.set_major_formatter(xMajorFormatter)
    ax1.yaxis.set_major_locator(yMajorLocator)
    ax1.yaxis.set_minor_locator(yMinorLocator)
    ax1.yaxis.set_major_formatter(yMajorFormatter)

    plt.xlim(2e-5, 20)

    # ylim + 6 will be shown
    ax1.set_ylim(1e-25, 2e-8)
    ax2.set_ylim(np.min(Td/T_nu)*0.5, np.max(Td/T_nu)*1.5)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    # plt.savefig(f'dens_evo_{load_str.replace("./", "").replace(".dat","")}_final.pdf')
    plt.show()

directory_in_str = './'
load_str_list = []

pathlist = Path(directory_in_str).glob('**/*.dat')
for path in pathlist:
    # because path is object not string
    load_str = str(path)
    if load_str[0] == 'm':
        load_str_list.append(load_str)


root = tkint.Tk()
def select(e):
    load_str = e.widget.get(*e.widget.curselection())

    plot_pandemic(load_str)


lstbox = tkint.Listbox(root, width=85, height=40)
lstbox.pack(padx=10, pady=10, fill='both', expand=True)

for i in load_str_list:
    # redefinition of i not needed if directory_in_str = './'
    lstbox.insert('end', i)

lstbox.bind('<<ListboxSelect>>',select) # Or lstbox.bind('<Double-1>',select) for double click

root.mainloop()