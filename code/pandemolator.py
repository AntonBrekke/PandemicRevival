#! /usr/bin/env python3

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root
from math import sqrt, log, log10, exp
import utils
import constants_functions as cf
import densities as dens

from scipy.special import kn

t_max = 1e16 / cf.hbar

rtol_ode = 1e-6
rtol_ode_pan = 1e-5
rtol_int = 1e-4

fac_abund_stop = 100.
xi_ratio_stop = 100.

class TimeTempRelation(object):
    def __init__(self, T_start=1e8, t_end=t_max, t_gp_pd=1e3, m_psi=None, dof_psi=None, k_psi=None):
        # print('Initializing TimeTempRelation')
        if m_psi is None:
            self.psi_in_SM = True
        else:
            self.psi_in_SM = False
            self.m_psi   = m_psi   # in GeV
            self.dof_psi = dof_psi
            self.k_psi   = k_psi   # +1 if psi is fermion, -1 for boson

        t_start = 1./(2.*self.hubble_of_temps(T_start, T_start))
        grid_size_time = int( log10(t_end/t_start) * t_gp_pd )
        # print(f'Time grid size: {grid_size_time}')
        self.t_grid = np.logspace(log10(t_start), log10(t_end), num=grid_size_time)
        self.sqrt_t_grid = np.sqrt(self.t_grid)

        sol = solve_ivp(self.der, [self.t_grid[0], self.t_grid[-1]], [T_start*self.sqrt_t_grid[0], T_start*self.sqrt_t_grid[0]], t_eval=self.t_grid, rtol=rtol_ode, atol=0.)
        self.T_SM_grid = sol.y[0]/self.sqrt_t_grid
        self.T_nu_grid = sol.y[1]/self.sqrt_t_grid
        self.hubble_grid = np.array([self.hubble_of_temps(T_SM, T_nu) for T_SM, T_nu in zip(self.T_SM_grid, self.T_nu_grid)])
        self.hubble_cumsimp = utils.cumsimp(self.t_grid, self.hubble_grid)
        self.sf_grid = np.exp(self.hubble_cumsimp)
        self.nu_dec_grid = (self.hubble_grid/(self.T_SM_grid**5.) > np.ones(grid_size_time)*cf.hubble_T5_nu_dec)
        self.dTSM_dt_grid = np.array([self.dTSM_dt(T_SM, hubble, nu_dec) for T_SM, hubble, nu_dec in zip(self.T_SM_grid, self.hubble_grid, self.nu_dec_grid)])
        self.dTnu_dt_grid = np.array([self.dTnu_dt(T_nu, hubble, nu_dec) for T_nu, hubble, nu_dec in zip(self.T_nu_grid, self.hubble_grid, self.nu_dec_grid)])

    def rho_psi(self, T_SM):
        if self.psi_in_SM:
            return 0.
        return cf.rho_boson(T_SM, self.m_psi, self.dof_psi) if self.k_psi == -1 else cf.rho_fermion(T_SM, self.m_psi, self.dof_psi)

    def P_psi(self, T_SM):
        if self.psi_in_SM:
            return 0.
        return cf.P_boson(T_SM, self.m_psi, self.dof_psi) if self.k_psi == -1 else cf.P_fermion(T_SM, self.m_psi, self.dof_psi)

    def rho_der_psi(self, T_SM):
        if self.psi_in_SM:
            return 0.
        return cf.rho_der_boson(T_SM, self.m_psi, self.dof_psi) if self.k_psi == -1 else cf.rho_der_fermion(T_SM, self.m_psi, self.dof_psi)

    def rho(self, T_SM, T_nu):
        return cf.rho_SM_no_nu(T_SM) + cf.rho_nu(T_nu) + cf.rho_m(T_SM, T_nu) + self.rho_psi(T_SM)

    def hubble_of_temps(self, T_SM, T_nu):
        return sqrt(8.*np.pi*cf.G*self.rho(T_SM, T_nu)/3.)

    def dTSM_dt(self, T_SM, hubble, nu_dec):
        if not nu_dec:
            return -3.*hubble*(cf.rho_SM_before_nu_dec(T_SM)+self.rho_psi(T_SM)+cf.P_SM_before_nu_dec(T_SM)+self.P_psi(T_SM))/(cf.rho_der_SM_before_nu_dec(T_SM)+self.rho_der_psi(T_SM))
        return -3.*hubble*(cf.rho_SM_no_nu(T_SM)+self.rho_psi(T_SM)+cf.P_SM_no_nu(T_SM)+self.P_psi(T_SM))/(cf.rho_der_SM_no_nu(T_SM)+self.rho_der_psi(T_SM))

    def dTnu_dt(self, T_nu, hubble, nu_dec):
        if not nu_dec:
            return self.dTSM_dt(T_nu, hubble, nu_dec)
        return -hubble*T_nu

    def der(self, t, Ts):
        t = t
        sqrt_t = sqrt(t)
        T_SM = Ts[0]/sqrt_t
        T_nu = Ts[1]/sqrt_t

        hubble = self.hubble_of_temps(T_SM, T_nu)
        hubble_T5 = hubble/(T_SM**5.)
        nu_dec = True if not np.isfinite(hubble_T5) or hubble_T5 > cf.hubble_T5_nu_dec else False

        # if nu_dec:
        #     print(T_nu)

        der_SM = T_SM/(2.*sqrt_t) + sqrt_t*self.dTSM_dt(T_SM, hubble, nu_dec)
        der_nu = T_nu/(2.*sqrt_t) + sqrt_t*self.dTnu_dt(T_nu, hubble, nu_dec)

        return [der_SM, der_nu]

class Pandemolator(object):
    def __init__(self, m_N1, m_N2, m_X, m_h, m_psi, k_d, k_X, k_psi, dof_d, dof_X, C_n, C_rho, C_xi0, t_grid, T_grid, dT_dt_grid, ent_grid, hubble_grid, sf_grid, i_ic, n_ic, rho_ic, i_end):
        self.m_N1 = m_N1 # in GeV
        self.k_N1 = k_d
        self.dof_N1 = dof_d

        self.m_N2 = m_N2 # in GeV
        self.k_N2 = k_d
        self.dof_N2 = dof_d

        self.m_chi = m_N1
        self.k_chi = k_d
        self.dof_chi = dof_d

        self.m_X = m_X # in GeV
        self.k_X = k_X # +1 for fermion, -1 for boson, 0 for Maxwell-Boltzmann
        self.dof_X = dof_X

        self.m_h = m_h

        # psi = e.g. SM neutrino
        self.m_psi = m_psi # in GeV
        self.k_psi = k_psi # +1 for fermion, -1 for boson, 0 for Maxwell-Boltzmann

        self.fac_n_X = 2. if self.m_X > self.m_N1 + self.m_N2 else 1.

        self.C_n = C_n     # rhs of Boltzmann-eq. for n_chi + self.fac_n_X*n_X
        self.C_rho = C_rho # rhs of Boltzmann-eq. for rho_chi + rho_X + rho_h
        self.C_xi0 = C_xi0 # part of rhs of Boltzmann-eq. for n setting xi = 0 (i.e. chi chi -> X X)

        self.t_grid = t_grid
        self.T_grid = T_grid
        self.dT_dt_grid = dT_dt_grid
        self.ent_grid = ent_grid
        self.hubble_grid = hubble_grid

        self.i_ic = i_ic
        self.n_ic = n_ic     # initial condition for n_chi + self.fac_n_X*n_X
        self.rho_ic = rho_ic # initial condition for rho_chi + rho_X
        # self.sf_grid = sf_grid / sf_grid[self.i_ic]
        self.sf_grid = (ent_grid[i_ic] / ent_grid)**(1./3.)
        self.i_end = i_end

        self.log_t_grid = np.log(self.t_grid)
        self.T_interp = utils.LogInterp(self.t_grid, self.T_grid)
        self.dT_dt_interp = utils.LogInterp(self.t_grid, -self.dT_dt_grid)
        self.ent_interp = utils.LogInterp(self.t_grid, self.ent_grid)
        self.H_interp = utils.LogInterp(self.t_grid, self.hubble_grid)

        self.t_interp_T_fc = interp1d(np.log(self.T_grid[::-1]), np.log(self.t_grid[::-1]), kind='linear')
        self.dT_dt_interp_T_fc = interp1d(np.log(self.T_grid[::-1]), np.log(-self.dT_dt_grid[::-1]), kind='linear')
        self.ent_interp_T_fc = interp1d(np.log(self.T_grid[::-1]), np.log(self.ent_grid[::-1]), kind='linear')
        self.H_interp_T_fc = interp1d(np.log(self.T_grid[::-1]), np.log(self.hubble_grid[::-1]), kind='linear')
        self.sf_interp_T_fc = interp1d(np.log(self.T_grid[::-1]), np.log(self.sf_grid[::-1]), kind='linear')
        self.t_interp_T = lambda T: np.exp(self.t_interp_T_fc(np.log(T)))
        self.dT_dt_interp_T = lambda T: -np.exp(self.dT_dt_interp_T_fc(np.log(T)))
        self.ent_interp_T = lambda T: np.exp(self.ent_interp_T_fc(np.log(T)))
        self.H_interp_T = lambda T: np.exp(self.H_interp_T_fc(np.log(T)))
        self.sf_interp_T = lambda T: np.exp(self.sf_interp_T_fc(np.log(T)))

    # Anton: Define necessary quantities. 
    # Watch out for factor of 2 form n_N1 + n_N2 + 2 * n_X
    # and rho_N1 + rho_N2 + rho_X
    def n_chi(self, T_chi, xi_chi):
        return dens.n(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi)

    def n_X(self, T_chi, xi_X):
        return dens.n(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def rho(self, T_chi, xi_chi, xi_X):
        return 2*dens.rho(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi) + dens.rho(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def P(self, T_chi, xi_chi, xi_X):
        return 2*dens.P(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi) + dens.P(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def rho_3P_diff(self, T_chi, xi_chi, xi_X):
        return 2*dens.rho_3P_diff(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi) + dens.rho_3P_diff(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def n_chi_der_T(self, T_chi, xi_chi):
        return dens.n_der_T(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi)

    def n_chi_der_xi(self, T_chi, xi_chi):
        return dens.n_der_xi(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi)

    def n_X_der_T(self, T_chi, xi_X):
        return dens.n_der_T(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)
    
    def n_X_der_xi(self, T_chi, xi_X):
        return dens.n_der_xi(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def rho_der_T(self, T_chi, xi_chi, xi_X):
        return 2*dens.rho_der_T(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi) + dens.rho_der_T(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    def rho_chi_der_xi(self, T_chi, xi_chi):
        return dens.rho_der_xi(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi)

    def rho_X_der_xi(self, T_chi, xi_X):
        return dens.rho_der_xi(self.k_X, T_chi, self.m_X, self.dof_X, xi_X)

    # Anton: solve root-equations for n, rho with jacobian factor 
    def n_rho_root(self, Txi_chi, n_in, rho_in):
        T_chi = exp(max(min(Txi_chi[0], 10.), -100.))
        xi_chi = min(Txi_chi[1]+self.m_chi/T_chi, (1.-1e-14)*self.m_X/(self.fac_n_X*T_chi))
        n = max(2*self.n_chi(T_chi, xi_chi) + self.fac_n_X*self.n_X(T_chi, self.fac_n_X*xi_chi), 1e-300)
        rho = max(self.rho(T_chi, xi_chi, self.fac_n_X*xi_chi), 1e-300)
        return [log(n/n_in), log(rho/rho_in)]

    def jac_n_rho_root(self, Txi_chi, n_in, rho_in):
        T_chi = exp(max(min(Txi_chi[0], 10.), -100.))
        xi_chi = min(Txi_chi[1]+self.m_chi/T_chi, (1.-1e-14)*self.m_X/(self.fac_n_X*T_chi))

        n = max(2*self.n_chi(T_chi, xi_chi) + self.fac_n_X*self.n_X(T_chi, self.fac_n_X*xi_chi), 1e-300)
        rho = max(self.rho(T_chi, xi_chi, self.fac_n_X*xi_chi), 1e-300)

        n_der_T = 2*self.n_chi_der_T(T_chi, xi_chi) + self.fac_n_X*self.n_X_der_T(T_chi, self.fac_n_X*xi_chi)
        rho_der_T = self.rho_der_T(T_chi, xi_chi, self.fac_n_X*xi_chi)
        
        n_der_xi = 2*self.n_chi_der_xi(T_chi, xi_chi) + self.fac_n_X*self.fac_n_X*self.n_X_der_xi(T_chi, self.fac_n_X*xi_chi)
        rho_der_xi = self.rho_chi_der_xi(T_chi, xi_chi) + self.fac_n_X*self.rho_X_der_xi(T_chi, self.fac_n_X*xi_chi)
        return [[T_chi*(n_der_T/n - (self.m_chi/(T_chi*T_chi))*n_der_xi/n), n_der_xi/n],
         [T_chi*(rho_der_T/rho - (self.m_chi/(T_chi*T_chi))*rho_der_xi/rho), rho_der_xi/rho]]

    # Anton: In case of xi = 0, only solve root-equation for rho with jacobian
    def rho_root(self, log_T_chi, rho_in):
        T_chi = exp(max(min(log_T_chi[0], 10.), -100.))
        rho = max(self.rho(T_chi, 0., 0.), 1e-300)
        return [log(rho/rho_in)]

    def jac_rho_root(self, log_T_chi, rho_in):
        T_chi = exp(max(min(log_T_chi[0], 10.), -100.))
        rho = max(self.rho(T_chi, 0., 0.), 1e-300)
        rho_der_T = self.rho_der_T(T_chi, 0., 0.)

        return [T_chi*rho_der_T/rho]

    # Anton: RHS of Boltzmann equation
    def der(self, log_x, y):
        x = exp(log_x)
        T = self.m_chi / x
        print('Temp.:', x)
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        ent = self.ent_interp_T(T)
        sf = self.sf_interp_T(T)
        Y = y[0]
        n = Y * ent
        rho = y[1] / (sf**4.)

        if rho < (1.+1e-10)*self.m_chi*n or n < 0.: # energy density too small
            return [0., 0.]
        elif rho/n - self.m_chi < self.m_chi:
            self.T_chi_last = (rho/n - self.m_chi)/1.5
            self.xi_chi_last = log(n/(self.dof_chi*((self.m_chi*self.T_chi_last/(2.*np.pi))**1.5)))+self.m_chi/self.T_chi_last
        root_sol = root(self.n_rho_root, [log(self.T_chi_last), self.xi_chi_last-self.m_chi/self.T_chi_last], args=(n, rho), jac=self.jac_n_rho_root, method='lm')
        T_chi = exp(root_sol.x[0])
        xi_chi = min(root_sol.x[1] + self.m_chi/T_chi, (1.-1e-14)*self.m_X/(self.fac_n_X*T_chi))
        xi_X = self.fac_n_X*xi_chi

        # print(np.exp(xi_chi), self.n_X(T_chi, self.fac_n_X*xi_chi)/self.n_chi(T_chi, xi_chi), dens.rho(self.k_X, T_chi, self.m_X, self.dof_X, 2.*xi_chi)/dens.rho(self.k_chi, T_chi, self.m_chi, self.dof_chi, xi_chi))
        # n_sol = 2*self.n_chi(T_chi, xi_chi) + self.fac_n_X*self.n_X(T_chi, xi_X)
        # rho_sol = self.rho(T_chi, xi_chi, xi_X)
        self.T_chi_last, self.xi_chi_last = T_chi, xi_chi
        # print(self.m_chi/T, T, self.m_chi/T_chi, Y * cf.s0 * self.m_chi / cf.rho_crit0_h2, rho, xi_chi, xi_X)

        # P = self.P(T_chi, xi_chi, xi_X, xi_h)

        C_n = self.C_n(T, T_chi, xi_chi, xi_X)
        C_rho = self.C_rho(T, T_chi, xi_chi, xi_X)
        # print('Rel. density changes:', C_n/(3.*H*n), C_rho/(3.*H*(rho+P)))

        der_Y = -(T/dT_dt)*C_n/ent
        der_rho = -(T/dT_dt)*(H*self.rho_3P_diff(T_chi, xi_chi, xi_X) + C_rho)*(sf**4.)

        return [der_Y, der_rho]

    # Anton: function tracking whenever xi effectively becomes zero
    def event_xi_zero(self, log_x, y):
        x = exp(log_x)
        T = self.m_chi / x
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        ent = self.ent_interp_T(T)
        sf = self.sf_interp_T(T)
        Y = y[0]
        n = Y * ent
        rho = y[1] / (sf**4.)

        if rho < (1.+1e-10)*self.m_chi*n: # energy density too small
            return 1.
        elif rho/n - self.m_chi < self.m_chi:
            # print(rho/n - self.m_chi)
            self.T_chi_last = (rho/n - self.m_chi)/1.5
            self.xi_chi_last = log(n/(self.dof_chi*((self.m_chi*self.T_chi_last/(2.*np.pi))**1.5)))+self.m_chi/self.T_chi_last
        # print('xi_zero', log(self.T_chi_last), self.xi_chi_last-self.m_chi/self.T_chi_last)
        root_sol = root(self.n_rho_root, [log(self.T_chi_last), self.xi_chi_last-self.m_chi/self.T_chi_last], args = (n, rho), jac=self.jac_n_rho_root, method='lm')
        T_chi = exp(root_sol.x[0])
        xi_chi = min(root_sol.x[1] + self.m_chi/T_chi, (1.-1e-14)*self.m_X/(self.fac_n_X*T_chi))
        xi_X = self.fac_n_X*xi_chi

        C_xi0 = self.C_xi0(T, T_chi, xi_chi, xi_X)
        # print(C_xi0/(H*n))
        return 1. - C_xi0/(5.*xi_ratio_stop*H*n)

    # Anton: Tracks if simulated abundance already exceeds observed abundance 
    def event_abund_large(self, log_x, y):
        x = exp(log_x)
        T = self.m_chi / x
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        ent = self.ent_interp_T(T)
        sf = self.sf_interp_T(T)
        Y = y[0]
        Odh2_today = Y * cf.s0 * self.m_chi / cf.rho_crit0_h2
        return 1. - Odh2_today/(fac_abund_stop*cf.omega_d0)

    def der_xi_0(self, log_x, y):
        x = exp(log_x)
        T = self.m_chi / x
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        sf = self.sf_interp_T(T)
        rho = y[0] / (sf**4.)
        root_sol = root(self.rho_root, [log(self.T_chi_last)], jac=self.jac_rho_root, args=(rho))
        T_chi = exp(root_sol.x[0])
        self.T_chi_last, self.xi_chi_last = T_chi, 0.
        # print(self.m_chi/T, T, T_chi)

        # P = self.P(T_chi, 0., 0.)

        C_rho = self.C_rho(T, T_chi, 0., 0.)
        # print('Rel. density changes:', C_rho/(3.*H*(rho+P)))

        der_rho = -(T/dT_dt)*(H*self.rho_3P_diff(T_chi, 0., 0.) + C_rho)*(sf**4.)

        return [der_rho]

    # Anton: Tracks whenever xi effectively is non-zero
    def event_xi_nonzero(self, log_x, y):
        # return 1. # if only considering xi = 0
        x = exp(log_x)
        T = self.m_chi / x
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        sf = self.sf_interp_T(T)
        rho = y[0] / (sf**4.)
        root_sol = root(self.rho_root, [log(self.T_chi_last)], jac=self.jac_rho_root, args=(rho))
        T_chi = exp(root_sol.x[0])
        self.T_chi_last, self.xi_chi_last = T_chi, 0.

        C_xi0 = self.C_xi0(T, T_chi, 0., 0.)
        n_chi = self.n_chi(T_chi, 0.)
        n_X = self.n_X(T_chi, 0.)
        # print(C_xi0, H*(n_chi + self.fac_n_X*n_X))

        # return C_xi0/(xi_ratio_stop*H*(n_chi + 2.*n_X + 2.*n_h)) - 1.
        return C_xi0/(xi_ratio_stop*H*(2*n_chi + 2.*n_X)) - 1.

    # Anton: Tracks if simulated abundance already exceeds observed abundance in xi=0 regime 
    def event_abund_large_xi_0(self, log_x, y):
        x = exp(log_x)
        T = self.m_chi / x
        H = self.H_interp_T(T)
        dT_dt = self.dT_dt_interp_T(T)
        sf = self.sf_interp_T(T)
        ent = self.ent_interp_T(T)
        rho = y[0] / (sf**4.)
        root_sol = root(self.rho_root, [log(self.T_chi_last)], jac=self.jac_rho_root, args = (rho))
        T_chi = exp(root_sol.x[0])
        self.T_chi_last, self.xi_chi_last = T_chi, 0.
        Y = (2*self.n_chi(T_chi, 0.) + self.fac_n_X*self.n_X(T_chi, 0.)) / ent
        Odh2_today = Y * cf.s0 * self.m_chi / cf.rho_crit0_h2
        return 1. - Odh2_today/(fac_abund_stop*cf.omega_d0)

    # Anton: Main solver 
    def pandemolate(self):
        """
        Anton: Not entirely clear how this works. We use the fact that the dark sector is 
        in equilibrium to get T_d, xi_d. As we have to unknown variables, we must solve two equations. 
        We solve for n, rho numerically to get n_num, rho_num. Using equilibirum, the analytical 
        expression is known. Hence, we solve 
        n_an(T_d, xi_d) = n_num, rho_an(T_d, xi_d) = rho_num
        using root-solvers to obtain T_d, xi_d. 
        In the special case of xi_d = 0, the system simplifies to only one variable, in which we solve
        for rho. 
        """
        dof_fac_chi = self.dof_chi if self.k_chi == -1 else self.dof_chi*7./8.
        dof_fac_X = self.dof_X if self.k_X == -1 else self.dof_X*7./8.
        self.T_chi_last = (self.rho_ic / (cf.pi2*(2*dof_fac_chi+dof_fac_X)/30.))**0.25
        self.xi_chi_last = 0.

        self.log_x_pts = np.log(self.m_chi/self.T_grid[self.i_ic:self.i_end+1])
        n_pts = self.log_x_pts.size

        self.t_grid_sol = self.t_grid[self.i_ic:self.i_end+1]
        self.T_grid_sol = self.T_grid[self.i_ic:self.i_end+1]
        self.H_grid_sol = self.hubble_grid[self.i_ic:self.i_end+1]
        self.T_chi_grid_sol = np.empty(n_pts)
        self.xi_chi_grid_sol = np.empty(n_pts)
        self.xi_X_grid_sol = np.empty(n_pts)
        self.n_chi_grid_sol = np.empty(n_pts)
        self.n_X_grid_sol = np.empty(n_pts)

        i_max = 0
        n0 = self.n_ic
        ent0 = self.ent_interp_T(self.T_grid[self.i_ic + i_max])
        rho0 = self.rho_ic
        sf0 = self.sf_grid[self.i_ic + i_max]
        # print("Enter pandemolate while-loop ")
        while i_max < n_pts - 1:
            # print(f'Pandemolator while loop iteration i_max={i_max}')
            if i_max > 0:#self.event_xi_nonzero(self.log_x_pts[i_max], [rho0*(sf0**4.)]) > 0.: # xi = 0 at beginning of calculation
                print(f'i_max > 0')
                def event_xi(log_x, y):
                    return self.event_xi_nonzero(log_x, y)
                event_xi.terminal = True
                event_xi.direction = -1
                def event_abund(log_x, y):
                    return self.event_abund_large_xi_0(log_x, y)
                event_abund.terminal = True
                event_abund.direction = -1
                print(f'Start solve_ivp xi zero')
                sol_xi0 = solve_ivp(self.der_xi_0, [self.log_x_pts[i_max], self.log_x_pts[-1]], [rho0*(sf0**4.)], t_eval=self.log_x_pts[i_max:], events=(event_xi, event_abund), rtol=rtol_ode_pan, atol=0., method='RK45', first_step=self.log_x_pts[i_max+1]-self.log_x_pts[i_max])
                print(f'End solve_ivp xi zero')
                i_xi_nonzero = i_max + sol_xi0.t.size - 1

                self.T_chi_last = (rho0 / (cf.pi2*(2*dof_fac_chi+dof_fac_X)/30.))**0.25
                self.xi_chi_last = 0.
                for i in range(i_max, i_xi_nonzero + 1):
                    ent = self.ent_interp_T(self.T_grid_sol[i])
                    sf = self.sf_interp_T(self.T_grid_sol[i])
                    rho = sol_xi0.y[0, i-i_max]/(sf**4.)
                    root_sol = root(self.rho_root, [log(self.T_chi_last)], jac=self.jac_rho_root, args=(rho))
                    self.T_chi_grid_sol[i] = exp(root_sol.x[0])
                    self.xi_chi_grid_sol[i] = 0.
                    self.T_chi_last, self.xi_chi_last = self.T_chi_grid_sol[i], self.xi_chi_grid_sol[i]
                    self.xi_X_grid_sol[i] = self.fac_n_X*self.xi_chi_grid_sol[i]
                    self.n_chi_grid_sol[i] = self.n_chi(self.T_chi_grid_sol[i], self.xi_chi_grid_sol[i])
                    self.n_X_grid_sol[i] = self.n_X(self.T_chi_grid_sol[i], self.xi_X_grid_sol[i])

                if sol_xi0.t_events[1].size == 0: # abundance always < fac_abund_stop*DM abundance
                    sf0 = self.sf_grid[self.i_ic + i_xi_nonzero]
                    rho0 = sol_xi0.y[0,-1]/(sf0**4.)
                    n0 = 2*self.n_chi(self.T_chi_last, 0.) + self.fac_n_X*self.n_X(self.T_chi_last, 0.)
                    ent0 = self.ent_grid[self.i_ic + i_xi_nonzero]
                    sf0 = self.sf_grid[self.i_ic + i_xi_nonzero]
                else: # abundance becomes > fac_abund_stop*DM abundance, loop ends then due to new n_pts
                    self.i_end = self.i_ic + i_xi_nonzero
                    n_pts = i_xi_nonzero + 1
            else:
                i_xi_nonzero = i_max
            if i_xi_nonzero < n_pts - 1:
                y0 = [n0/ent0, rho0*(sf0**4.)]
                def event_xi(log_x, y):
                    return self.event_xi_zero(log_x, y)
                event_xi.terminal = True
                event_xi.direction = -1
                def event_abund(log_x, y):
                    return self.event_abund_large(log_x, y)
                event_abund.terminal = True
                event_abund.direction = -1
                # print('Start solve_ivp for Y, rho')
                print('Start solve_ivp xi non-zero')
                # print(self.log_x_pts[i_xi_nonzero], self.log_x_pts[-1])
                sol = solve_ivp(self.der, [self.log_x_pts[i_xi_nonzero], self.log_x_pts[-1]], y0, t_eval=self.log_x_pts[i_xi_nonzero:], events=(event_xi, event_abund), rtol=rtol_ode_pan, atol=0., method='RK45', first_step=self.log_x_pts[i_xi_nonzero+1]-self.log_x_pts[i_xi_nonzero])
                print('End solve_ivp xi non-zero')
                i_max = i_xi_nonzero + sol.t.size - 1

                self.T_chi_last = (rho0 / (cf.pi2*(2*dof_fac_chi+dof_fac_X)/30.))**0.25
                self.xi_chi_last = 0.
                i_start = i_xi_nonzero + 1 if i_xi_nonzero > 0 else 0
                for i in range(i_start, i_max + 1):
                    ent = self.ent_interp_T(self.T_grid_sol[i])
                    sf = self.sf_interp_T(self.T_grid_sol[i])
                    n = sol.y[0, i-i_xi_nonzero]*ent
                    rho = sol.y[1, i-i_xi_nonzero]/(sf**4.)
                    root_sol = root(self.n_rho_root, [log(self.T_chi_last), (self.xi_chi_last-self.m_chi/self.T_chi_last)], jac=self.jac_n_rho_root, args=(n, rho), method='lm')
                    # print(exp(root_sol.x[0]))
                    self.T_chi_grid_sol[i] = exp(root_sol.x[0])
                    self.xi_chi_grid_sol[i] = min(root_sol.x[1] + self.m_chi/self.T_chi_grid_sol[i], (1.-1e-14)*self.m_X/(self.fac_n_X*self.T_chi_grid_sol[i]))#root_sol.x[1] + self.m_chi/self.T_chi_grid_sol[i]
                    self.T_chi_last, self.xi_chi_last = self.T_chi_grid_sol[i], self.xi_chi_grid_sol[i]
                    self.xi_X_grid_sol[i] = self.fac_n_X*self.xi_chi_grid_sol[i]
                    self.n_chi_grid_sol[i] = self.n_chi(self.T_chi_grid_sol[i], self.xi_chi_grid_sol[i])
                    self.n_X_grid_sol[i] = self.n_X(self.T_chi_grid_sol[i], self.xi_X_grid_sol[i])

                ent0 = self.ent_interp_T(self.T_grid_sol[i_max])
                sf0 = self.sf_interp_T(self.T_grid_sol[i_max])
                n0 = sol.y[0,-1]*ent0
                rho0 = sol.y[1,-1]/(sf0**4.)
                self.T_chi_last = self.T_chi_grid_sol[i_max]
                self.xi_chi_last = self.xi_chi_grid_sol[i_max]

                if sol.t_events[1].size != 0 or sol.t.size < 2: # abundance becomes > fac_abund_stop*DM abundance, loop ends then due to new n_pts
                    self.i_end = self.i_ic + sol.t.size + i_xi_nonzero - 1
                    n_pts = i_max + 1
            else:
                i_max = i_xi_nonzero

        # print("Exit pandemolate while-loop ")
        # shorten solution grids, relevant if integration stopped due to large abundance
        self.t_grid_sol = self.t_grid_sol[:n_pts]
        self.T_grid_sol = self.T_grid_sol[:n_pts]
        self.H_grid_sol = self.H_grid_sol[:n_pts]
        self.T_chi_grid_sol = self.T_chi_grid_sol[:n_pts]
        self.xi_chi_grid_sol = self.xi_chi_grid_sol[:n_pts]
        self.xi_X_grid_sol = self.xi_X_grid_sol[:n_pts]
        self.n_chi_grid_sol = self.n_chi_grid_sol[:n_pts]
        self.n_X_grid_sol = self.n_X_grid_sol[:n_pts]

if __name__ == '__main__':
    from math import asin, cos, sin
    import C_res_scalar
    import C_res_vector
    import matplotlib.pyplot as plt

    m_d = 1e-4
    m_a = 0.
    m_X = 2.5e-4
    sin2_2th = 1e-16
    y = 1e-4

    k_d = 1.
    k_a = 1.
    k_X = -1.
    dof_d = 2.
    dof_X = 3.

    m_d2 = m_d*m_d
    m_a2 = m_a*m_a
    m_X2 = m_X*m_X
    th = 0.5*asin(sqrt(sin2_2th))
    c_th = cos(th)
    s_th = sin(th)
    y2 = y*y

    # Anton: For some reason, matrix elements are added here
    # M2_dd = 2. * y2 * (c_th**4.) * (m_X2 - 4.*m_d2)
    # M2_aa = 2. * y2 * (s_th**4.) * (m_X2 - 4.*m_a2)
    # M2_da = 2. * y2 * (s_th**2.) * (c_th**2.) * (m_X2 - ((m_a+m_d)**2.))

    # M2_X23 = 2*g**2/m_X^2 * (m_X2 - (m2 - m3)**2)*(2*m_X2 + (m2 + m3)**2)
    # New matrix elements for X --> 23
    # M2_dd = 2.*y2*(c_th**4.)/m_X2 * (m_X2)*(2*m_X2 + (2*m_d)**2)
    # M2_aa = 2.*y2*(s_th**4.)/m_X2 * (m_X2)*(2*m_X2)
    # M2_da = 2.*y2*(s_th**2.)*(c_th**2.)/m_X2 * (m_X2 - m_d**2)*(2*m_X2 + m_d**2)

    M2_dd = 4*y2*(c_th**4.)*(m_X2-6*m_d2)
    M2_da = 4*y2*(s_th**2.)*(c_th**2.)*(m_X2-m_d2)
    M2_aa = 4.*y2*(s_th**4.)*m_X2

    def C_n(T_a, T_d, xi_d, xi_X):
        C_pp_dd = C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, y2*y2*(c_th**8.)) / 4. # symmetry factor 1/4
        C_da = C_res_vector.C_n_3_12(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d,   0., xi_X, M2_da)
        C_aa = C_res_vector.C_n_3_12(m_a, m_a, m_X, k_a, k_a, k_X, T_a, T_a, T_d,   0.,   0., xi_X, M2_aa) / 2.
        print('C_ns:', C_pp_dd, C_da, C_aa)
        return C_da + 2.*C_aa + 2.*C_pp_dd
    def C_rho(T_a, T_d, xi_d, xi_X):
        C_da = C_res_vector.C_rho_3_12(2, m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_da)
        C_aa = C_res_vector.C_rho_3_12(3, m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d,   0., 0., xi_X, M2_aa) / 2. # symmetry factor 1/2
        return C_da + C_aa
    def C_xi0(T_a, T_d, xi_d, xi_X):
        C_pp_dd = C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, y2*y2*(c_th**8.), type=1) / 4.
        return 2.*C_pp_dd

    Ttrel = TimeTempRelation()
    ent_grid = np.array([cf.s_SM_no_nu(T)+cf.s_nu(T_nu) for T, T_nu in zip(Ttrel.T_SM_grid, Ttrel.T_nu_grid)])
    T_d_dw = cf.T_d_dw(m_d) # temperature of maximal d production by Dodelson-Widrow mechanism
    i_ic = np.argmax(Ttrel.T_nu_grid < T_d_dw)
    i_end = np.argmax(Ttrel.T_nu_grid < 0.01*m_d)
    sf_ic_norm_0 = (cf.s0/(cf.s_SM_no_nu(Ttrel.T_SM_grid[i_ic]) + cf.s_nu(Ttrel.T_nu_grid[i_ic])))**(1./3.)
    n_ic = cf.n_0_dw(m_d, th) / (sf_ic_norm_0**3.)
    rho_ic = n_ic * cf.avg_mom_0_dw(m_d) / sf_ic_norm_0
    pan = Pandemolator(m_d, k_d, dof_d, m_X, k_X, dof_X, m_a, k_a, C_n, C_rho, C_xi0, Ttrel.t_grid, Ttrel.T_nu_grid, Ttrel.dTnu_dt_grid, ent_grid, Ttrel.hubble_grid, Ttrel.sf_grid, i_ic, n_ic, rho_ic, i_end)

    T_d_last = m_d/2363419.2747363993
    xi_d_last = 2363419.529806943
    n = 3.270460613423846e-23
    rho = 3.486289072693418e-27
    root_sol = root(pan.n_rho_root, [log(T_d_last), xi_d_last-m_d/T_d_last], args = (n, rho), jac=pan.jac_n_rho_root, method='lm')
    T_d = exp(root_sol.x[0])
    xi_d = min(root_sol.x[1] + m_d/T_d, (1.-1e-14)*0.5*m_X/T_d)
    print(T_d, xi_d)
    # exit(1)

    pan.pandemolate()

    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], pan.n_chi_grid_sol/ent_grid[i_ic:i_end+1], color='dodgerblue')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], pan.n_X_grid_sol/ent_grid[i_ic:i_end+1], color='darkorange')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], (pan.n_chi_grid_sol+2.*pan.n_X_grid_sol)/ent_grid[i_ic:i_end+1], color='mediumorchid')
    plt.show()

    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], pan.T_chi_grid_sol*np.abs(pan.xi_chi_grid_sol), color='dodgerblue')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], pan.T_chi_grid_sol*np.abs(0.5*pan.xi_X_grid_sol), color='darkorange')
    plt.show()

    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1], pan.T_chi_grid_sol, color='dodgerblue')
    plt.show()

    i_skip = 100
    C_dd = np.array([-C_res_vector.C_n(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_dd, type=-1) / 2. for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_inv_dd = np.array([C_res_vector.C_n(m_d, m_d, m_X, k_d, k_d, k_X, T_d, T_d, T_d, xi_d, xi_d, xi_X, M2_dd, type=1) / 2. for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_da = np.array([-C_res_vector.C_n(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_da, type=-1) for T_d, T_a, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.T_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_inv_da = np.array([C_res_vector.C_n(m_d, m_a, m_X, k_d, k_a, k_X, T_d, T_a, T_d, xi_d, 0., xi_X, M2_da, type=1) for T_d, T_a, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.T_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_aa = np.array([-C_res_vector.C_n(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_da, type=-1) / 2. for T_d, T_a, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.T_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_inv_aa = np.array([C_res_vector.C_n(m_a, m_a, m_X, k_d, k_a, k_X, T_a, T_a, T_d, 0., 0., xi_X, M2_da, type=1) / 2. for T_d, T_a, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.T_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_ann = np.array([-C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, y2*y2*(c_th**8.), type=-1) / 4. for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    C_inv_ann = np.array([C_res_vector.C_n_XX_dd(m_d, m_X, k_d, k_X, T_d, xi_d, xi_X, y2*y2*(c_th**8.), type=1) / 4. for T_d, xi_d, xi_X in zip(pan.T_chi_grid_sol[::i_skip], pan.xi_chi_grid_sol[::i_skip], pan.xi_X_grid_sol[::i_skip])])
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_dd, color='dodgerblue')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_inv_dd, color='dodgerblue', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_da, color='darkorange')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_inv_da, color='darkorange', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_aa, color='yellowgreen')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_inv_aa, color='yellowgreen', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_ann, color='tomato')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_inv_ann, color='tomato', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 3.*Ttrel.hubble_grid[i_ic:i_end+1:i_skip]*pan.n_chi_grid_sol[::i_skip], color='mediumorchid')
    plt.show()

    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_dec, color='dodgerblue')
    # plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], C_inv_dec, color='darkorange', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_ann, color='tomato')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 2.*C_inv_ann, color='yellowgreen', ls='--')
    plt.loglog(m_d/Ttrel.T_nu_grid[i_ic:i_end+1:i_skip], 3.*Ttrel.hubble_grid[i_ic:i_end+1:i_skip]*pan.n_X_grid_sol[::i_skip], color='mediumorchid')
    plt.show()
