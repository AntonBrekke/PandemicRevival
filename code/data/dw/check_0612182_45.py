#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

fig4 = np.loadtxt('0612182_dw_fig_4.dat', skiprows=2)
fig5 = np.loadtxt('0612182_dw_fig_5.dat', skiprows=2)

plt.loglog(fig4[:,0], fig4[:,1], color='dodgerblue')
plt.loglog(fig5[:,0], 4e-8/fig5[:,1], color='yellowgreen')
plt.show()
