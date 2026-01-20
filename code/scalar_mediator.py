#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import sin, cos, sqrt, atan, log
from scipy.integrate import quad

rtol_int = 1e-4

# @nb.jit(nopython=True, cache=True)
def Gamma_phi(y, th, m_phi, m_d, m_X):
    y2 = y*y
    sth = np.sin(th)
    cth = np.cos(th)
    m_h = m_phi

    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    m_h2 = m_h*m_h
    """
    Anton: 
    M2_X->23 = 4*g^2*((gV^2+1)*(m1-m2)*(m1+m2)+4*(gV^2-1)*m2*m3-((gV^2+1)*m3^2))
    gV=1 (0) for vector (no vector) coupling in Feynman-rule 
    i*g_23 * gamma^\mu * (g_V - gamma^5)
    gV = 0 for 2=a, 3=a or 2=d, 3=d
    gV = 1 for 2=a, 3=d or 2=d, 3=a
    Gamma_X_23 = |p_f|/(2^M*8*pi*m_X2)*|M_X->23| * H(m_X - (m2 + m3))
    |p_f| = 1/(2*m1)*sqrt(m1^4 + m2^4 + m3^4 - 2*m1^2*m2^2 - 2*m1^2*m3^2 - 2*m2^2*m3^2)
    |p_f| = 1/(2*m_X)*sqrt((m_X2 - m2^2 - m3^2)^2 - 4*m2^2*m3^2)
    """
    
    M2_h_dd = 2*(4*y2*m_d2/m_X2)*(cth**4)*(m_h2-4*m_d2)
    M2_h_da = 2*(4*y2*m_d2/m_X2)*(cth**2)*(sth**2)*(m_h2-m_d2)
    M2_h_aa = 2*(4*y2*m_d2/m_X2)*(sth**4)*m_h2
    # y --> y*m_X/(2*m_d)
    M2_h_XX = 4*y2*(m_h2**2-4*m_h2*m_X2+12*m_X2**2)/m_X2

    pf_dd = 1/(2*m_h)*np.sqrt(((m_h2 - 2*m_d2)**2 - 4*m_d2*m_d2)*(m_h > 2*m_d))
    pf_ad = 1/(2*m_h)*(m_h2 - m_d2) 
    pf_aa = m_h/2
    pf_XX = 1/(2*m_h)*np.sqrt(((m_h2 - 2*m_X2)**2 - 4*m_X2*m_X2)*(m_h > 2*m_X))

    # Anton: Decay to aa, ad, and dd. Have used m_a = 0.
    Gamma_h_dd = pf_dd/(16*np.pi*m_h2) * M2_h_dd * (m_h > 2*m_d)
    Gamma_h_da = pf_ad/(8*np.pi*m_h2) * M2_h_da * (m_h > m_d)
    Gamma_h_aa = pf_aa/(16*np.pi*m_h2) * M2_h_aa * (m_h > 0)
    Gamma_h_XX = pf_XX/(16*np.pi*m_h2) * M2_h_XX * (m_h > 2*m_X)

    return Gamma_h_aa + Gamma_h_da + Gamma_h_dd + Gamma_h_XX

# sub indicates if s-channel on-shell resonance is subtracted
# @nb.jit(nopython=True, cache=True)
def M2_gen(s, t, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=False):
    m12 = m1*m1
    m22 = m2*m2
    m32 = m3*m3
    m42 = m4*m4
    m13 = m12*m1
    m23 = m22*m2
    m33 = m32*m3
    m43 = m42*m4

    u = m12 + m22 + m32 + m42 - s - t

    """
    Anton: 
    https://arxiv.org/pdf/2309.16615
    Subtract on-shell contribution from Breit-Wigner propagator (RIS-subtraction) to 
    avoid double counting decay processes
    D_BW(s) = 1 / (s - m^2 + imG) = (s - m^2)/((s-m^2)^2 + (mG)^2) - imG/((s-m^2)^2 + (mG)^2)
    |D_BW(s)|^2 = 1 / ((s-m^2)^2 + (mG)^2) := s_prop
    The real part of D_BW is defined as the off-shell propagator 
    D_off-shell(s) := Re(D_BW(s)) -- used for interference terms st, su
    D_off-shell(t) := Re(D_BW(t)) -- used for interference term st
    D_off-shell(u) := Re(D_BW(u)) -- used for interference term su
    Need another expression for squared propagator:
    |D_off-shell(s)|^2 := ((s - m^2)^2 - (mG)^2) / ((s-m^2)^2 + (mG)^2)^2 -- used in ss
    """
    # Anton: Squared BW-propagators, |D_BW|^2
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)

    ss = ((m1+m2)*(m1+m2)-s)*((m3+m4)*(m3+m4)-s)*s_prop*(s_prop*((s-m_phi2)*(s-m_phi2) - m_Gamma_phi2) if sub else 1.)
    tt = ((m1+m3)*(m1+m3)-t)*((m2+m4)*(m2+m4)-t)*t_prop
    uu = ((m1+m4)*(m1+m4)-u)*((m2+m3)*(m2+m3)-u)*u_prop
    st = -(m23*m3+m13*m4+m22*m3*(m3+m4)+m12*m4*(m2+m3+m4)-s*(m3*m4+t)
     +m1*(m22*m3+m3*m42+m43+m2*(m32+2.*m3*m4+m42-s)-m4*s-m3*t-m4*t)
     +m2*(m33+m32*m4-m4*t-m3*(s+t)))*s_prop*t_prop*((s-m_phi2)*(t-m_phi2)+(0. if sub else m_Gamma_phi2))
    su = -(m13*m3+m23*m4+m22*m4*(m3+m4)+m12*m3*(m2+m3+m4)-s*(m3*m4+u)
     +m1*(m33+m22*m4+m32*m4+m2*(m32+2.*m3*m4+m42-s)-m4*u-m3*(s+u))
     +m2*(m3*m42+m43-m4*s-m3*u-m4*u))*s_prop*u_prop*((s-m_phi2)*(u-m_phi2)+(0. if sub else m_Gamma_phi2))
    tu = -((m13*m2+m33*m4+m42*m3*(m3+m4)+m12*m2*(m2+m3+m4)-m3*m4*(t+u)-t*u
     +m1*(m23+m32*m4+m3*m42+m22*(m3+m4)-m3*t+m2*(2.*m3*m4-t-u)-m4*u)
     +m2*(m32*m4+m3*m42-m4*t-m3*u))
     *((t-m_phi2)*(u-m_phi2)+m_Gamma_phi2))*t_prop*u_prop

    return 4.*vert*ss#(ss+tt+uu+st+su+tu)

# @nb.jit(nopython=True, cache=True)
def M2_gen_ss(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2):
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    ss = (((m1+m2)**2.)-s)*(((m3+m4)**2.)-s)*s_prop
    return 4.*vert*ss

# @nb.jit(nopython=True, cache=True)
def M2_fi(s, t, m_d2, vert, m_phi2, m_Gamma_phi2):
    u = 2.*m_d2 - s - t

    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)

    ss = s * (s-4.*m_d2) * s_prop
    tt = (t - m_d2) * (t - m_d2) * t_prop
    uu = (u - m_d2) * (u - m_d2) * u_prop
    st = s * (t + m_d2) * ((s-m_phi2)*(t-m_phi2) + m_Gamma_phi2) * s_prop * t_prop
    su = s * (u + m_d2) * ((s-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * s_prop * u_prop
    tu = - (3.*m_d2*m_d2 - t*u - m_d2*(t+u)) * ((t-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * t_prop * u_prop

    return 4.*vert*(ss + tt + uu + st + su + tu)

# @nb.jit(nopython=True, cache=True)
def M2_tr(s, t, m_d2, vert, m_phi2, m_Gamma_phi2):
    u = 3.*m_d2 - s - t
    s_prop = 1. / ((s-m_phi2)*(s-m_phi2) + m_Gamma_phi2)
    t_prop = 1. / ((t-m_phi2)*(t-m_phi2) + m_Gamma_phi2)
    u_prop = 1. / ((u-m_phi2)*(u-m_phi2) + m_Gamma_phi2)
    ss = (4.*m_d2*m_d2 - 5.*m_d2*s + s*s) * s_prop
    tt = (4.*m_d2*m_d2 - 5.*m_d2*t + t*t) * t_prop
    uu = (4.*m_d2*m_d2 - 5.*m_d2*u + u*u) * u_prop
    st = - (5.*m_d2*m_d2 - s*t - 2.*m_d2*(s+t)) * ((s-m_phi2)*(t-m_phi2) + m_Gamma_phi2) * s_prop * t_prop
    su = - (5.*m_d2*m_d2 - s*u - 2.*m_d2*(s+u)) * ((s-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * s_prop * u_prop
    tu = - (5.*m_d2*m_d2 - t*u - 2.*m_d2*(t+u)) * ((t-m_phi2)*(u-m_phi2) + m_Gamma_phi2) * t_prop * u_prop

    return 4.*vert*(ss + tt + uu + st + su + tu)

# @nb.jit(nopython=True, cache=True)
def M2_el(s, t, m_d, vert, m_phi2, m_Gamma_phi2):
    return M2_gen(s, t, m_d, m_d, m_d, m_d, vert, m_phi2, m_Gamma_phi2)

# @nb.jit(nopython=True, cache=True)
def ker_sigma_gen(t, s, p1cm, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub):
    return M2_gen(s, t, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=sub)/(64.*np.pi*s*p1cm*p1cm)

# no factor taking care of identical particles (not known on this level)
def sigma_gen(s, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub=False):
    if s < (m1+m2)**2. or s < (m3+m4)**2.:
        return 0.
    sqrt_s = sqrt(s)
    E1cm = (s+m1*m1-m2*m2)/(2.*sqrt_s)
    p1cm = sqrt((E1cm-m1)*(E1cm+m1))
    E3cm = (s+m3*m3-m4*m4)/(2.*sqrt_s)
    p3cm = sqrt((E3cm-m3)*(E3cm+m3))

    a = (m1*m1-m3*m3-m2*m2+m4*m4)/(2.*sqrt_s)
    t0 = (a-(p1cm-p3cm))*(a+(p1cm-p3cm))
    t1 = (a-(p1cm+p3cm))*(a+(p1cm+p3cm))
    # t0 = ((m1*m1-m3*m3-m2*m2+m4*m4)**2.)/(4.*s) - ((p1cm-p3cm)**2.)
    # t1 = ((m1*m1-m3*m3-m2*m2+m4*m4)**2.)/(4.*s) - ((p1cm+p3cm)**2.)

    res, err = quad(ker_sigma_gen, t1, t0, args=(s, p1cm, m1, m2, m3, m4, vert, m_phi2, m_Gamma_phi2, sub), epsabs=0., epsrel=rtol_int)

    return res

# @nb.jit(nopython=True, cache=True)
def sigma_tr(s, m_d2, vert, m_phi2, m_Gamma_phi2):
    if s < 4.*m_d2:
        return 0.
    if s > 1e6*m_d2 and s > 1e6*m_phi2:
        return 0.5*1.5*vert/(np.pi*s) # factor 0.5 due to identical particles in final state
    m_phi4 = m_phi2 * m_phi2
    m_d4 = m_d2 * m_d2
    s2 = s*s
    m_Gamma_phi = sqrt(m_Gamma_phi2)
    sqrt_fac = sqrt(s*(s-4.*m_d2))

    fac_atan = 2.*(-3.*m_d2+2.*m_phi2+s)*((4.*m_d4-5.*m_d2*m_phi2+m_phi4)*(s-m_phi2)*(s-m_phi2)
     -m_Gamma_phi2*(m_d4+m_d2*m_phi2+m_phi4-6.*m_phi2*s+3.*s2)-2.*m_Gamma_phi2*m_Gamma_phi2)/m_Gamma_phi
    sum_atan = fac_atan*atan((m_Gamma_phi*sqrt_fac*(s-m_d2))/(m_d4*m_d2-3.*m_d2*m_phi2*s+m_Gamma_phi2*s+m_phi4*s+m_phi2*s2))

    fac_log = (-5.*m_phi4*m_phi4 + 15.*m_d4*m_d2*(m_phi2 - s) - 4.*m_phi2*s2*s
     + m_phi4*(9.*s*m_phi2 - 4.*m_Gamma_phi2) - 7.*s*m_phi2*m_Gamma_phi2 + m_Gamma_phi2*m_Gamma_phi2
     - m_d4*(30.*m_phi4 + 3.*s2 - 33.*s*m_phi2 + 8.*m_Gamma_phi2)
     + m_d2*(23.*m_phi4*m_phi2 + s2*s + 12.*s2*m_phi2 + 4.*s*m_Gamma_phi2
     - 36.*s*m_phi4 + 15.*m_phi2*m_Gamma_phi2))
    sum_log = fac_log * log(
     (-2.*m_d4*m_d2+2.*m_phi4*s+s2*s+s2*sqrt_fac+3.*m_d4*(3.*s+sqrt_fac)
      -2.*m_d2*(3.*s2+2.*s*sqrt_fac+m_phi2*(3.*s+sqrt_fac))+2.*(m_phi2*s*sqrt_fac+m_phi2*s2+m_Gamma_phi2)) /
     (-2.*m_d4*m_d2+2.*m_phi4*s+s2*s-s2*sqrt_fac+3.*m_d4*(3.*s-sqrt_fac)
      +2.*m_d2*(-3.*s2+2.*s*sqrt_fac+m_phi2*(-3.*s+sqrt_fac))+2.*(-m_phi2*s*sqrt_fac+m_phi2*s2+m_Gamma_phi2)))

    sum_3 = (m_d2-s)*(3.*m_d2-2.*m_phi2-s)*(4.*m_d4+3.*m_phi4+6.*s2-m_d2*(s+4.*m_phi2)-8.*s*m_phi2+3.*m_Gamma_phi2)*sqrt_fac/s

    # factor 0.5 due to identical particles in final state
    return 0.5*vert*(sum_atan+sum_log+sum_3)/(4.*np.pi*(s-m_d2)*(s-m_d2)*(s-3.*m_d2+2.*m_phi2)*((s-m_phi2)*(s-m_phi2)+m_Gamma_phi2))

# @nb.jit(nopython=True, cache=True)
def sigma_el(s, m_d2, vert, m_phi2, m_Gamma_phi2):
    if s < 4.*m_d2:
        return 0.
    if s > 1e6*m_d2 and s > 1e6*m_phi2:
        return 0.5*1.5*vert/(np.pi*s) # factor 0.5 due to identical particles in final state
    m_phi4 = m_phi2 * m_phi2
    m_d4 = m_d2 * m_d2
    s2 = s*s
    m_Gamma_phi = sqrt(m_Gamma_phi2)

    fac_atan = 2.*(-4.*m_d2+2.*m_phi2+s)*(-(((m_phi2-4.*m_d2)*(s-m_phi2))**2.)+m_Gamma_phi2*(m_phi4-6.*m_phi2*s+3.*s2)+2.*m_Gamma_phi2*m_Gamma_phi2)
    sum_atan = fac_atan*atan(m_Gamma_phi*(4.*m_d2-s)/(-4.*m_d2*m_phi2+m_phi4+s*m_phi2+m_Gamma_phi2))

    fac_log = m_Gamma_phi*(5.*m_phi4*m_phi4+4.*m_phi2*s2*s+64.*m_d4*m_d2*(s-m_phi2)+16.*m_d4*(5.*m_phi4-5.*s*m_phi2+m_Gamma_phi2)
     +m_phi4*(-9.*m_phi2*s+4.*m_Gamma_phi2)+7.*s*m_phi2*m_Gamma_phi2-m_Gamma_phi2*m_Gamma_phi2
     -4.*m_d2*(9.*m_phi4*m_phi2+s*(4.*s*m_phi2+m_Gamma_phi2)+m_phi2*(-13.*s*m_phi2+5.*m_Gamma_phi2)))
    sum_log = fac_log*log((m_phi4+m_Gamma_phi2)/(m_Gamma_phi2+((s-4.*m_d2+m_phi2)**2.)))

    sum_3 = m_Gamma_phi*(4.*m_d2-s)*(4.*m_d2-2.*m_phi2-s)*(16.*m_d4-8.*m_d2*m_phi2+3.*m_phi4+6.*s2-8.*s*m_phi2+3.*m_Gamma_phi2)

    # factor 0.5 due to identical particles in final state
    return 0.5*vert*(sum_atan+sum_log+sum_3)/(4.*np.pi*m_Gamma_phi*s*(s-4.*m_d2)*(s-4.*m_d2+2.*m_phi2)*(m_phi4+s2-2.*s*m_phi2+m_Gamma_phi2))

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    m_d = 1e-5      # GeV 
    m_a = 0.
    m_phi = 5*m_d
    m_X = 3*m_d
    sin2_2th = 1e-12
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    y = 2e-4

    # fi = aa->dd, tr = ad->dd, el = ss->ss
    # For small theta << 1, sin^2(theta) = 1/4*sin^2(2*theta)
    vert_fi = y**4 * np.cos(th)**4*np.sin(th)**4
    vert_tr = y**4 * np.cos(th)**6*np.sin(th)**2
    vert_el = y**4 * np.cos(th)**8

    th_arr = np.linspace(0, 2*np.pi, 1000)
    Gamma = Gamma_phi(y=y*(2*m_phi/m_X), th=th_arr, m_phi=m_phi, m_d=m_d, m_X=m_X)
    plt.plot(th_arr, Gamma)
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], labels=[r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$', r'$2\pi$'])
    plt.show()

    m1 = m_a
    m2 = m_d
    s_min = (m1 + m2)**2
    S = 10**(np.linspace(np.log10(s_min), 0.1, int(1e3)))
    # S = np.linspace(s_min, 1e2, int(1e3))
    T = np.linspace(-1, 1, int(1e3))
    s, t = np.meshgrid(S, T, indexing='ij')

    Gamma = Gamma_phi(y=y, th=th, m_phi=m_phi, m_d=m_d)
    m_Gamma_phi2 = (m_phi*Gamma)**2
    print(Gamma)

    sigma = np.vectorize(sigma_gen)(s=S, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2, sub=False)
    sigma_elast = np.vectorize(sigma_el)(s=S, m_d2=m_d**2, vert=vert_el, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)
    sigma_trans = np.vectorize(sigma_tr)(s=S, m_d2=m_d**2, vert=vert_tr, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)

    plt.loglog(S, sigma, 'r--', label='sigma_gen|tr')
    # plt.loglog(S, 2*sigma_el, 'g', label='sigma_el')
    plt.loglog(S, sigma_trans, 'g', label='sigma_tr')
    plt.legend()
    plt.show()

    # 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    M2_trans = M2_tr(s, t, m_d2=m_d**2, vert=vert_tr, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)
    # M2_general_tHoft = M2_gen_tHoft(s, t, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)
    M2_elast = np.vectorize(M2_el)(s, t, m_d=m_d, vert=vert_el, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)
    plot_M2 = ax1.contourf(s, t, np.log10(M2_elast), levels=300, cmap='jet')
    fig1.colorbar(plot_M2)
    ax1.set_xscale('log')

    # 2 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    M2_general = M2_gen(s, t, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_phi2=m_phi**2, m_Gamma_phi2=m_Gamma_phi2)
    # They are equal up to ~ r decimals, where abs((a-b)/b) = C*10^(-r)
    # print(-np.log10(np.max(abs((M2_general_tHoft - M2_general)/M2_general))))
    plot_M2_gen = ax2.contourf(s, t, np.log10(M2_general), levels=300, cmap='jet')
    cbar2 = fig2.colorbar(plot_M2_gen)
    # plt.plot(S[::10], np.zeros_like(S[::10]), 'ko')
    ax2.set_xscale('log')

    # Can make colorbar interactive - see constant value that you click
    from matplotlib import colors
    highlight_cmap = colors.ListedColormap(['k'])
    highlight = ax2.imshow(np.ma.masked_all_like(np.log10(M2_general)), interpolation='nearest', vmin=np.log10(M2_general).min(), vmax=np.log10(M2_general).max(), extent=[S.min(),S.max(),T.min(), T.max()], cmap=highlight_cmap, origin='lower', aspect='auto', zorder=10)

    # highlight = [ax2.contour(s, t, (M2_general), colors='none')]

    def on_pick(event):
        val = event.mouseevent.ydata
        selection = np.ma.masked_outside(np.log10(M2_general), val-0.2, val+0.2)
        highlight.set_data(selection.T)
        # highlight[0].remove()
        # highlight[0] = ax2.contour(s, t, selection, colors='k')
        fig2.canvas.draw()
    cbar2.ax.set_picker(5)
    fig2.canvas.mpl_connect('pick_event', on_pick)

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()