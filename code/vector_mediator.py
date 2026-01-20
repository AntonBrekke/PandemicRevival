#! /usr/bin/env python3

import numpy as np
import numba as nb
from math import sin, cos, sqrt, atan, log
from scipy.integrate import quad
from scalar_mediator import Gamma_phi
import time

rtol_int = 1e-4

"""
Anton:
Changes:
Updated 
* Gamma_phi to Gamma_X 
* M2_tr
* M2_gen
* M2_el
* M2_fi
* sigma_gen
* M2_gen_ss

What must be done: 
Update 
sigma_tr, sigma_el

However, only sigma_gen is ever called in the code - 
may not have to fix the others.. 
sigma_el appears in "C_res_scalar_no_spin_stat"

el: dd --> dd
tr: da --> dd
fi: aa --> dd
gen: 12 --> 34, 1,2,3,4 = {d,a}

Implemented real intermediate state (RIS) subtraction
"""

# Total decay-rate
@nb.jit(nopython=True, cache=True)
def Gamma_X_new(y, m_X, m_N1, m_N2, m0, m12, m2, ma):

    g12 = y
    g1nu = y * m2*ma/(m0*m12)

    m_X2 = m_X*m_X
    m_N12 = m_N1*m_N1
    m_N22 = m_N2*m_N2
    """
    Anton: 
    Gamma_X_23 = |p_f|/(2^M*8*pi*m_X2)*|M_X->23| * H(m_X - (m2 + m3))
    |p_f| = 1/(2*m1)*sqrt(m1^4 + m2^4 + m3^4 - 2*m1^2*m2^2 - 2*m1^2*m3^2 - 2*m2^2*m3^2)
    |p_f| = 1/(2*m1)*sqrt((m1^2 - m2^2 - m3^2)^2 - 4*m2^2*m3^2)
    """
    
    M2_12 = 2.*g12**2 * (m_X+m_N1-m_N2)*(m_X-m_N1+m_N2)*(2*m_X2 + (m_N1+m_N2)**2)
    M2_1nu = 2.*g1nu**2 * (m_X - m_N1)*(m_X + m_N1)*(2*m_X2 + m_N12)

    pf_12 = 1/(2*m_X)*sqrt((m_X2 - m_N12 - m_N22)**2 - 4*m_N12*m_N22)
    pf_1nu = 1/(2*m_X)*(m_X2 - m_N12)

    # Anton: Decay to aa, ad, and dd. Have used m_a = 0.
    Gamma_X_12 = pf_12/(16*np.pi*m_X2) * M2_12 * (m_X > m_N1 + m_N2)
    Gamma_X_1nu = pf_1nu/(16*np.pi*m_X2) * M2_1nu * (m_X > m_N1)
    # Gamma_X_10 is not here as m_X > m0 + m_N1 will not happen. 

    return Gamma_X_12 + Gamma_X_1nu

# Total decay-rate
@nb.jit(nopython=True, cache=True)
def Gamma_X(y, th, m_X, m_d):
    y2 = y*y
    sth = np.sin(th)
    cth = np.cos(th)

    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    """
    Anton: 
    M2_X->23 = 2g^2/m_X^2 * (m_X^2 - (m2 - m3)^2) * [2*m_X^2 + (m2 + m3)^2]
    Gamma_X_23 = |p_f|/(2^M*8*pi*m_X2)*|M_X->23| * H(m_X - (m2 + m3))
    |p_f| = 1/(2*m_X)*sqrt((m_X2 - m2^2 - m3^2)^2 - 4*m2^2*m3^2) 
    """
    M2_aa = 2.*y2*(sth**4.)/m_X2 * (m_X2)*(2*m_X2)
    M2_ad = 2.*y2*(sth**2.)*(cth**2.)/m_X2 * (m_X2 - m_d2)*(2*m_X2 + m_d2)
    M2_dd = 2.*y2*(cth**4.)/m_X2 * (m_X2)*(2*m_X2 + 4*m_d2)

    pf_aa = m_X/2 
    pf_ad = 1/(2*m_X)*(m_X2 - m_d2) 
    pf_dd = 1/(2*m_X)*sqrt((m_X2 - 2*m_d2)**2 - 4*m_d2*m_d2) 

    # Anton: Decay to aa, ad, and dd. Have used m_a = 0.
    X_aa = pf_aa/(16*np.pi*m_X2) * M2_aa * (m_X > 0)
    X_ad = pf_ad/(8*np.pi*m_X2) * M2_ad * (m_X > m_d)
    X_dd = pf_dd/(16*np.pi*m_X2) * M2_dd * (m_X > 2*m_d)

    return X_aa + X_ad + X_dd

#####################################
# Anton: Matrix elements for gen, tr, fi, el

# sub indicates if s-channel on-shell resonance is subtracted
# Anton: NB! Open at own risk.
# Anton: Include scalar Higgs diagrams
@nb.jit(nopython=True, cache=True)
def M2_gen_new_3(s, t, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub=False):
    """
    Anton: 
    12 --> 34, 1,2,3,4 = a, d
    sub = True: |D_off-shell|^2 is used -- on-shell contribution is subtracted
    sub = False: |D_BW|^2 is used
    """
    m12 = m1*m1
    m13 = m1*m12
    m14 = m12*m12
    m15 = m12*m13
    m16 = m12*m14
    # m17 = m13*m14
    m18 = m14*m14

    m22 = m2*m2
    m23 = m2*m22
    m24 = m22*m22
    m25 = m22*m23
    m26 = m22*m24
    # m27 = m23*m24
    m28 = m24*m24

    m32 = m3*m3
    m33 = m3*m32
    m34 = m32*m32
    m35 = m32*m33
    m36 = m32*m34
    # m37 = m33*m34
    m38 = m34*m34

    m42 = m4*m4
    m43 = m4*m42
    m44 = m42*m42
    m45 = m42*m43
    m46 = m42*m44
    # m47 = m43*m44
    m48 = m44*m44

    m_d4 = m_d2*m_d2
    m_X4 = m_X2*m_X2
    u = m12 + m22 + m32 + m42 - s - t

    s2 = s*s
    s3 = s*s2
    t2 = t*t
    t3 = t*t2
    u2 = u*u
    u3 = u*u2

    m_Gamma_h = np.sqrt(m_Gamma_h2)
    m_Gamma_X = np.sqrt(m_Gamma_X2)

    """
    Anton: 
    https://arxiv.org/pdf/2309.16615
    Subtract on-shell contribution from Breit-Wigner propagator (RIS-subtraction) to 
    avoid double counting decay processes
    Goal: D_BW --> D_off-shell, |D_BW|^2 --> |D_off-shell|^2
    D_BW(s) = 1 / (s - m^2 + imG) = (s - m^2)/((s-m^2)^2 + (mG)^2) - imG/((s-m^2)^2 + (mG)^2)
    |D_BW(s)|^2 = 1 / ((s-m^2)^2 + (mG)^2) := s_prop
    D_BW(s) = (s-m^2)*s_prop - imG*s_prop = s_prop*((s-m^2)-imG)
    The real part of D_BW is defined as the off-shell propagator 
    D_off-shell(s) := Re(D_BW(s)) = (s-m^2)*s_prop -- used for interference terms st, su
    D_off-shell(t) := Re(D_BW(t)) = (t-m^2)*t_prop -- used for interference term st
    D_off-shell(u) := Re(D_BW(u)) = (u-m^2)*u_prop -- used for interference term su
    Need another expression for squared off-shell propagator:
    |D_off-shell(s)|^2 := ((s - m^2)^2 - (mG)^2) / ((s-m^2)^2 + (mG)^2)^2 = ((s-m^2)^2 - (mG)^2)*s_prop*s_prop
    -- used in ss
    Also recall: |M|^2 contains only 2*Re(D(s)* x D(t)) etc. for cross-terms, so only the real part.
    Re(D(s)* x D(t)) = s_prop*t_prop*((s - m_X2)*(t - m_X2) + (0 if sub else m_GammaX2))
    |D(s)|^2 = s_prop*s_prop*((s - m_X2)^2 - m_Gamma_X2 if sub else 1.) 
    EDIT: Numba does not like one-line if-else tests (like written above for propagator). Split 
    test into if-else blocks.
    Summary: 
    X, Y = s, t, u
    |D_BW(X)|^2 = X_prop
    D_BW(X) = X_prop*((X-m^2)-imG)
    Re(D_BW(X)* x D_BW(Y)) = X_prop*Y_prop*((X-mx^2)*(Y-my^2)+mxG*myG)
    |D_off-shell(X)|^2 = X_prop*X_prop*((X-m^2)^2 - (mG)^2)
    D_off-shell(X) = X_prop*(X-m^2)
    Re(D_off-shell(X)* x D_off_shell(Y)) = X_prop*Y_prop*(X-m^2)*(Y-m^2) = D_off-shell(X) x D_off_shell(Y)
    """

    # Anton: Squared BW-propagators for Dark Photon, |D_BW|^2
    s_propX = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_propX = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_propX = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)
    # Anton: Squared BW-propagators for Dark Higgs, |D_BW|^2
    s_proph = 1. / ((s-m_h2)*(s-m_h2) + m_Gamma_h2)
    t_proph = 1. / ((t-m_h2)*(t-m_h2) + m_Gamma_h2)
    u_proph = 1. / ((u-m_h2)*(u-m_h2) + m_Gamma_h2)

    # Subtract s-channel resonance (ignore t and u)
    if sub:
        # Off-shell propagators
        # D_off-shell(X) = X_prop * (X - m^2) 
        # sproph = s_proph*(s-m_h2)
        # tproph = t_proph*(t-m_h2)
        # uproph = u_proph*(u-m_h2)
        # spropX = s_propX*(s-m_X2)
        # tpropX = t_propX*(t-m_X2)
        # upropX = u_propX*(u-m_X2)
        # Squared off-shell propagators 
        # |D_off-shell(X)|^2 = X_prop*X_prop*((X-m^2)^2 - (mG)^2)
        # sproph2 = s_proph*s_proph*((s-m_h2)*(s-m_h2) - m_Gamma_h2)
        # tproph2 = t_proph*t_proph*((t-m_h2)*(t-m_h2) - m_Gamma_h2)
        # uproph2 = u_proph*u_proph*((u-m_h2)*(u-m_h2) - m_Gamma_h2)
        # spropX2 = s_propX*s_propX*((s-m_X2)*(s-m_X2) - m_Gamma_X2)
        # tpropX2 = t_propX*t_propX*((t-m_X2)*(t-m_X2) - m_Gamma_X2)
        # upropX2 = u_propX*u_propX*((u-m_X2)*(u-m_X2) - m_Gamma_X2)

        ssh = (64*m_d4*(m12+2*m2*m1+m22-s)*(m32+2*m4*m3+m42-s))/m_X4 * s_proph*s_proph*((s-m_h2)*(s-m_h2) - m_Gamma_h2)

        sth = (32*m_d4*((-2*m42-2*m3*m4+2*m2*(m3-m4)+s+t-u)*m12-2*((m3-m4)*m22+(m32+2*m4*m3+m42-s)*m2+m3*m42-m32*m4-m3*t+m4*u)*m1-s2-t2+u2+m32*s+m42*s+2*m3*m4*s+m32*t+m42*t+m22*(-2*m32-2*m4*m3+s+t-u)-m32*u-m42*u-2*m2*(m4*m32-m42*m3+u*m3-m4*t)))/m_X4 * s_proph*(s-m_h2)*t_proph*(t-m_h2)

        suh = (32*m_d4*((-2*m32-2*m2*m3-2*m4*m3+2*m2*m4+s-t+u)*m12+2*((m3-m4)*m22-(m32+2*m4*m3+m42-s)*m2+m3*m42-m32*m4-m3*t+m4*u)*m1-s2+t2-u2+m32*s+m42*s+2*m3*m4*s-m32*t-m42*t+m32*u+m42*u+m22*(-2*m42-2*m3*m4+s-t+u)+2*m2*(m4*m32-m42*m3+u*m3-m4*t)))/m_X4 * s_proph*(s-m_h2)*u_proph*(u-m_h2)

        ssX = (1/(m_X4))*4*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+2*m2*(s-2*m_X2)*m15+(2*m24+(2*m32+2*m42-8*m_X2+3*s-2*t-2*u)*m22+s2-t2-u2+m3*m4*(4*m_X2-2*s)+m42*s-2*m_X2*s+2*m42*t+2*m_X2*t-2*s*t+2*m42*u+2*m_X2*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))*m14-4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+((2*m32+2*m42-8*m_X2+3*s-2*t-2*u)*m24+2*(2*m_X4+2*s*m_X2+6*t*m_X2+6*u*m_X2-s2+t2+u2+2*m3*m4*(s-2*m_X2)-2*s*t+m32*(4*m42-8*m_X2+3*s-2*t-2*u)-2*s*u+2*t*u-m42*(8*m_X2-3*s+2*(t+u)))*m22+2*m42*m_X4-s3+m42*s2+2*m_X2*s2-2*m_X2*t2+s*t2-2*m_X2*u2+s*u2-2*m42*m_X2*s+2*m3*m4*(2*m_X4-2*s*m_X2+s2)-2*m_X4*t+2*m42*m_X2*t-2*m42*s*t-2*m_X4*u+6*m42*m_X2*u-2*m42*s*u-4*m_X2*t*u+2*s*t*u+m32*(2*m_X4+(-2*s+6*t+2*u)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))*m12+2*m2*((s-2*m_X2)*m24-2*(2*m_X2-s)*(m32+m42-t-u)*m22+2*m42*m_X4-s3+m42*s2+2*m_X2*s2-2*m_X2*t2+s*t2-2*m_X2*u2+s*u2-2*m_X4*s-2*m42*m_X2*s+2*m3*m4*(4*m_X4-2*s*m_X2+s2)+4*m42*m_X2*t-2*m42*s*t+4*m42*m_X2*u-2*m42*s*u-4*m_X2*t*u+2*s*t*u+m32*(2*m_X4+(4*(t+u)-2*s)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32-2*m4*s*m3+t2+u2-m42*(t+u))+m24*((-4*m42+s+2*(t+u))*m32+m4*(4*m_X2-2*s)*m3+s2-t2-u2-2*m_X2*s+2*m_X2*t-2*s*t+2*m_X2*u-2*s*u-2*t*u+m42*(s+2*(t+u)))+m22*(-2*t*m_X4-2*u*m_X4+2*s2*m_X2-2*t2*m_X2-2*u2*m_X2-4*t*u*m_X2-s3+s*t2+s*u2+2*m3*m4*(2*m_X4-2*s*m_X2+s2)+2*s*t*u+m42*(2*m_X4+(-2*s+6*t+2*u)*m_X2+s*(s-2*(t+u)))+m32*(2*m_X4+(-2*s+2*t+6*u)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))) * s_propX*s_propX*((s-m_X2)*(s-m_X2) - m_Gamma_X2)

        stX = (1/(m_X4))*2*(2*m18+(4*m22+4*m32+4*m42-3*s-3*t-5*u)*m16+2*(2*m23+2*(m3-m4)*m22+(2*m32-s-u)*m2+2*m33-2*m32*m4+m4*(4*m_X2+u)-m3*(t+u))*m15+(2*m24+4*(m3-m4)*m23+(4*m32-4*m4*m3+4*m42-2*(4*m_X2+s+t+3*u))*m22+2*(2*m33-2*m4*m32-(4*m_X2+u)*m3+m4*(t+u))*m2+2*m34+4*u2-4*m33*m4-3*m42*s-3*m42*t+4*s*t-5*m42*u+4*m_X2*u+4*s*u+4*t*u+2*m3*m4*(s+u)+m32*(4*m42-2*(4*m_X2+s+t+3*u)))*m14+(4*m25+4*(m3-m4)*m24+(8*m32+8*m42-4*(2*m_X2+s+t+2*u))*m23+2*(4*m33-4*m4*m32+4*m42*m3-2*(2*m_X2+s+t+2*u)*m3+m4*(4*m_X2+s+3*(t+u)))*m22+(4*m34+(8*m42-4*(2*m_X2+s+t+2*u))*m32+s2-t2+3*u2-4*m_X2*s+4*m_X2*t+4*s*t+4*m_X2*u+4*s*u+2*t*u-4*m42*(s+u))*m2+4*m35-4*m34*m4-2*m4*(4*m_X2+u)*(s+t+u)+2*m32*m4*(4*m_X2+3*s+t+3*u)+m3*(4*(s-t+u)*m_X2-s2-(4*m42-t-3*u)*(t+u)+2*s*(2*t+u))+m33*(8*m42-4*(2*m_X2+s+t+2*u)))*m13+(4*(m3-m4)*m25+(-8*m_X2-4*m3*m4+s+t-u)*m24+2*(4*m33-4*m4*m32+4*m42*m3-2*(s+t+2*u)*m3+m4*s+3*m4*(t+u))*m23+(-8*m4*m33+(4*m42-32*m_X2+3*(s+t-u))*m32+2*m4*(3*(s+t+u)-8*m_X2)*m3+m42*(-24*m_X2+s+t-u)+2*(4*m_X4+(5*s+5*t+7*u)*m_X2-s2-t2+u2))*m22+(4*m35-4*m4*m34+(8*m42-4*(s+t+2*u))*m33+2*m4*(3*(s+t+u)-8*m_X2)*m32-(-8*m_X4-8*u*m_X2+s2+t2-3*u2-4*s*t-2*s*u-2*t*u+4*m42*(4*m_X2+u))*m3+2*m4*(4*m_X4-2*(s-u)*m_X2-t2-2*t*u-u*(s+u)))*m2+s3+t3-u3-m42*s2-m42*t2-s*t2+m42*u2-4*m_X2*u2-s*u2-t*u2-4*m35*m4-s2*t+4*m42*s*t+m34*(-8*m_X2+s+t-u)-8*m_X4*u+12*m42*m_X2*u+s2*u+t2*u-4*m_X2*s*u-4*m_X2*t*u-2*s*t*u+2*m33*m4*(3*s+t+3*u)+2*m3*m4*(4*m_X4-2*(t-u)*m_X2-s2-2*s*u-u*(t+u))+m32*((-24*m_X2+s+t-u)*m42+2*(4*m_X4+(5*s+5*t+7*u)*m_X2-s2-t2+u2)))*m12-(2*(2*m32+4*m4*m3+2*m_X2-s+u)*m25+2*(2*m33+2*m4*m32+(2*m_X2-s-t+u)*m3+m4*(2*m_X2-t))*m24+(4*m34+16*m4*m33+2*(4*m_X2-3*s-3*t+u)*m32-8*m4*(s+t+u)*m3+3*s2+t2-3*u2-8*m_X2*t-8*m_X2*u-2*t*u+2*m42*(4*m_X2-s+t+u))*m23+(4*m35+4*m4*m34+2*(4*m_X2-3*s-3*t+u)*m33+2*m4*(8*m_X2-3*s-3*t-u)*m32+(-8*m_X4-4*(s+3*u)*m_X2+2*m42*(12*m_X2-s-t+u)+3*(s2+t2-u2))*m3+2*m4*(-4*m_X4+2*(s+t-u)*m_X2+t*(t+u)))*m22+(8*m4*m35+2*(2*m_X2-s-t+u)*m34-8*m4*(s+t+u)*m33+(-8*m_X4-4*(t+3*u)*m_X2+2*m42*(12*m_X2-s-t+u)+3*(s2+t2-u2))*m32+2*m4*(-16*m_X4+4*(s+t)*m_X2+s2+(t+u)**2+2*s*u)*m3-s3-t3+u3-4*m_X2*s2+4*m_X2*t2+s*t2+4*m_X2*u2+s*u2+t*u2+8*m_X4*s+s2*t-4*m_X2*s*t-s2*u-t2*u+4*m_X2*t*u+2*s*t*u+m42*(-8*m_X4+4*(s-2*u)*m_X2+s2+t2-u2-4*s*t))*m2+m34*m4*(4*m_X2-2*s)+2*m35*(2*m_X2-t+u)+2*m32*m4*(-4*m_X4+2*(s+t-u)*m_X2+s*(s+u))+m33*(2*(4*m_X2+s-t+u)*m42+s2+3*t2-3*u2-2*s*u-8*m_X2*(s+u))+4*m4*m_X2*((2*m_X2-t)*u-s*(2*t+u))+m3*(8*t*m_X4+4*(s2-t*s+u*s-t2+u2)*m_X2-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(-8*m_X4+4*(t-2*u)*m_X2+s2+t2-u2-4*s*t)))*m1+m25*(-4*m33-4*m4*m32+2*(2*m_X2+s-u)*m3+4*m4*m_X2)+m24*(-2*m34-4*m4*m33+(s+t-u)*m32+2*m4*(2*m_X2+t)*m3+2*m_X2*(-s+t+u))+2*m_X2*(2*m4*m35+(s-t+u)*m34-2*m4*(s+t+u)*m33+((4*m_X2+3*s-t+u)*m42-s2+t2-u2-4*m_X2*u-2*s*u)*m32+2*m4*(t*(s+u)-2*m_X2*s)*m3+4*m_X2*u*(u-m42))-m23*(4*m35+4*m4*m34+(-6*s-6*t+2*u)*m33-2*m4*(4*m_X2+3*s+t+u)*m32+(-2*(4*m_X2+s-t-u)*m42+3*s2+t2-3*u2-2*t*u+4*m_X2*(s+t+u))*m3+4*m4*m_X2*(s+t+u))+m22*(-4*m4*m35+(s+t-u)*m34+2*m4*(4*m_X2+s+3*t+u)*m33+((-16*m_X2+s+t-u)*m42-s2-t2+u2+12*m_X2*u)*m32+2*m4*(4*m_X4-2*(s+2*t)*m_X2-t*(t+u))*m3+2*m_X2*((4*m_X2-s+3*t+u)*m42+s2-t2-u2-4*m_X2*u-2*t*u))+m2*(2*(2*m_X2+t-u)*m35+2*m4*(2*m_X2+s)*m34+(2*(4*m_X2-s+t-u)*m42-s2-3*t2+3*u2+2*s*u-4*m_X2*(s+t+u))*m33+2*m4*(4*m_X4-2*(2*s+t)*m_X2-s*(s+u))*m32+(-8*u*m_X4+4*t*u*m_X2+4*s*(2*t+u)*m_X2+s3+t3-u3-t*u2-s*(t+u)**2+t2*u+s2*(u-t)+m42*(8*m_X4-8*(s+t)*m_X2-s2-t2+u2+4*s*t))*m3+4*m4*m_X2*(s*(t+u)-2*m_X2*t))) * s_propX*(s-m_X2)*t_propX*(t-m_X2)

        suX = (1/(m_X4))*2*(2*m18+(4*m22+4*m32+4*m42-3*s-5*t-3*u)*m16+2*(2*m23-2*(m3-m4)*m22+(2*m42-s-t)*m2+m3*(-2*m42+4*m_X2+t)+m4*(2*m42-t-u))*m15+(2*m24-4*(m3-m4)*m23+(4*m32-4*m4*m3+4*m42-2*(4*m_X2+s+3*t+u))*m22+(4*m43-2*(4*m_X2+t)*m4+2*m3*(-2*m42+t+u))*m2+2*m44+4*m_X2*t+2*m3*m4*(-2*m42+s+t)+m32*(4*m42-3*s-5*t-3*u)+4*(s+t)*(t+u)-2*m42*(4*m_X2+s+3*t+u))*m14+(4*m25-4*(m3-m4)*m24+(8*m32+8*m42-4*(2*m_X2+s+2*t+u))*m23+2*(4*m4*m32+(-4*m42+4*m_X2+s+3*t+3*u)*m3+2*m4*(2*m42-2*m_X2-s-2*t-u))*m22+(4*m44-4*(2*m_X2+s+2*t+u)*m42+s2+3*t2-u2-4*m_X2*s+4*m_X2*t+4*s*t+m32*(8*m42-4*(s+t))+4*m_X2*u+4*s*u+2*t*u)*m2+m32*(8*m43-4*m4*(t+u))+m4*(4*m44-4*(2*m_X2+s+2*t+u)*m42-s2+3*t2+u2+2*s*t+4*m_X2*(s+t-u)+4*s*u+4*t*u)-2*m3*(2*m44-(4*m_X2+3*s+3*t+u)*m42+(4*m_X2+t)*(s+t+u)))*m13+(-4*(m3-m4)*m25+(-8*m_X2-4*m3*m4+s-t+u)*m24+2*(4*m4*m32+(-4*m42+s+3*(t+u))*m3+2*m4*(2*m42-s-2*t-u))*m23+((4*m42-24*m_X2+s-t+u)*m32+2*m4*(-4*m42-8*m_X2+3*(s+t+u))*m3+m42*(3*(s-t+u)-32*m_X2)+2*(4*m_X4+(5*s+7*t+5*u)*m_X2-s2+t2-u2))*m22+(4*m4*(2*m42-4*m_X2-t)*m32-2*(2*m44+(8*m_X2-3*(s+t+u))*m42-4*m_X4+t2+u2+2*m_X2*(s-t)+s*t+2*t*u)*m3+m4*(4*m44-4*(s+2*t+u)*m42+8*m_X4-s2+3*t2-u2+8*m_X2*t+2*s*t+4*s*u+2*t*u))*m2+8*m42*m_X4+s3-t3+u3-8*m44*m_X2-2*m42*s2+2*m42*t2-4*m_X2*t2-s*t2-2*m42*u2-s*u2+t*u2+m44*s+10*m42*m_X2*s-m44*t-8*m_X4*t+14*m42*m_X2*t+s2*t-4*m_X2*s*t+m44*u+10*m42*m_X2*u-s2*u-t2*u-4*m_X2*t*u-2*s*t*u+m32*((-24*m_X2+s-t+u)*m42-s2+t2-u2+12*m_X2*t+4*s*u)-2*m3*m4*(2*m44-(3*s+3*t+u)*m42-4*m_X4+s2+t2+2*s*t-2*m_X2*(t-u)+t*u))*m12-(2*(2*m42+4*m3*m4+2*m_X2-s+t)*m25+2*(m3*(2*m42+2*m_X2-u)+m4*(2*m42+2*m_X2-s+t-u))*m24+(4*m44+2*(4*m_X2-3*s+t-3*u)*m42+8*m3*(2*m42-s-t-u)*m4+3*s2-3*t2+u2-8*m_X2*t-8*m_X2*u-2*t*u+2*m32*(4*m_X2-s+t+u))*m23+(2*m4*(12*m_X2-s+t-u)*m32+2*(2*m44+(8*m_X2-3*s-t-3*u)*m42-4*m_X4+2*m_X2*(s-t+u)+u*(t+u))*m3+m4*(4*m44+2*(4*m_X2-3*s+t-3*u)*m42-8*m_X4-4*m_X2*(s+3*t)+3*(s2-t2+u2)))*m22+(2*(2*m_X2-s+t-u)*m44-(8*m_X4+4*(3*t+u)*m_X2-3*(s2-t2+u2))*m42+2*m3*(4*m44-4*(s+t+u)*m42-16*m_X4+s2+t2+u2+2*s*t+2*t*u+4*m_X2*(s+u))*m4-s3+t3-u3-4*m_X2*s2+4*m_X2*t2+s*t2+4*m_X2*u2+s*u2-t*u2+8*m_X4*s-s2*t+s2*u+t2*u-4*m_X2*s*u+4*m_X2*t*u+2*s*t*u+m32*(-8*m_X4+4*(s-2*t)*m_X2+s2-t2+u2+2*m42*(12*m_X2-s+t-u)-4*s*u))*m2+m32*m4*(-8*m_X4+(4*u-8*t)*m_X2+s2-t2+u2+2*m42*(4*m_X2+s+t-u)-4*s*u)+m4*(2*(2*m_X2+t-u)*m44+(-8*(s+t)*m_X2+s2-3*t2+3*u2-2*s*t)*m42-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+8*m_X4*u+s2*u+t2*u+2*s*t*u+4*m_X2*(s2+t*s-u*s+t2-u2))+2*m3*((2*m_X2-s)*m44+(-4*m_X4+2*(s-t+u)*m_X2+s*(s+t))*m42+2*m_X2*(2*t*m_X2-t*u-s*(t+2*u))))*m1+m25*(2*m4*(-2*m42+2*m_X2+s-t)-4*m3*(m42-m_X2))+m24*(-2*m44+(s-t+u)*m42+2*m3*(-2*m42+2*m_X2+u)*m4+2*m_X2*(-s+t+u))+2*m_X2*((s+t-u)*m44-(4*t*m_X2+s2+t2-u2+2*s*t)*m42+2*m3*(m44-(s+t+u)*m42-2*m_X2*s+(s+t)*u)*m4+4*m_X2*t2+m32*(m42*(4*m_X2+3*s+t-u)-4*m_X2*t))+m23*(2*m4*(4*m_X2+s-t-u)*m32+(-4*m44+2*(4*m_X2+3*s+t+u)*m42-4*m_X2*(s+t+u))*m3-m4*(4*m44+(-6*s+2*t-6*u)*m42+3*s2-3*t2+u2-2*t*u+4*m_X2*(s+t+u)))+m22*((s-t+u)*m44-(-12*t*m_X2+s2-t2+u2)*m42-2*m3*(2*m44-(4*m_X2+s+t+3*u)*m42-4*m_X4+u*(t+u)+2*m_X2*(s+2*u))*m4-2*m_X2*(4*t*m_X2-s2+(t+u)**2)+m32*((-16*m_X2+s-t+u)*m42+2*m_X2*(4*m_X2-s+t+3*u)))+m2*(m4*(8*m_X4-8*(s+u)*m_X2-s2+t2-u2+4*s*u+2*m42*(4*m_X2-s-t+u))*m32+2*((2*m_X2+s)*m44+(4*m_X4-2*(2*s+u)*m_X2-s*(s+t))*m42-4*m_X4*u+2*m_X2*s*(t+u))*m3+m4*(2*(2*m_X2-t+u)*m44-(4*(s+t+u)*m_X2+s2-3*t2+3*u2-2*s*t)*m42+s3-t3+u3+t*u2-s*(t+u)**2-8*m_X4*t+s2*(t-u)-t2*u+4*m_X2*t*u+4*m_X2*s*(t+2*u)))) * s_propX*(s-m_X2)*u_propX*(u-m_X2)

        shtX = -(1/(m_X4))*8*m_d2*(2*(m3-m4)*m15+(4*m32+2*m2*(m3-m4)-s+t-u)*m14+2*(2*m33+2*(m42-s-u)*m3+2*m22*(m3-m4)+m4*(s+t+u))*m13+(4*m34+(8*m42-2*(2*m_X2+3*s+t+3*u))*m32+2*m4*(t-4*m_X2)*m3-4*m42*m_X2+s2-t2+u2-m42*s+4*m_X2*s+m42*t+2*m2*(2*m4*m32+(2*m42+2*m_X2-s-u)*m3+m4*(t-2*m_X2))+m22*(8*m32+4*m42-3*s+t-u)-3*m42*u+2*s*u)*m12+2*(m35+m4*m34+2*(m42-s-u)*m33+m4*(2*m_X2-s-u)*m32+((-2*m_X2-2*s+t-2*u)*m42+s2-t2+u2+2*m_X2*t+2*s*u)*m3-m2*(m32+2*m4*m3+m42-s)*(4*m_X2-t)+m22*(2*m33+2*m4*m32+(4*m42-2*m_X2-2*s+t-2*u)*m3+m4*(2*m_X2+t))-m4*(2*u*m_X2+s*t))*m1+m32*s2-4*m_X2*s2-m32*t2+m32*u2-m34*s-3*m32*m42*s+4*m32*m_X2*s+4*m42*m_X2*s+8*m3*m4*m_X2*s+m34*t+m32*m42*t-2*m3*m4*s*t+m22*((4*m42-4*m_X2-s+t-3*u)*m32+2*m4*(t-4*m_X2)*m3+4*m_X2*(s-m42))-m34*u-m32*m42*u+2*m32*s*u-2*m2*(m35+m4*m34+(2*m42-s-t-u)*m33+m4*(2*m_X2-t)*m32+(-((2*m_X2+t)*m42)+s*t+2*m_X2*u)*m3-2*m4*m_X2*t)) * t_propX*(t-m_X2)*(-s_proph)*(s-m_h2)

        shuX = (1/(m_X4))*8*m_d2*(2*(m3-m4)*m15+(-4*m42+2*m2*(m3-m4)+s+t-u)*m14+(4*m22*(m3-m4)-2*(2*m4*m32+(s+t+u)*m3+2*m4*(m42-s-t)))*m13-(4*m44-4*m_X2*m42-6*s*m42-6*t*m42-2*u*m42+2*m3*(u-4*m_X2)*m4+s2+t2-u2+4*m_X2*s+2*s*t+m32*(8*m42-4*m_X2-s-3*t+u)+m22*(4*m32+8*m42-3*s-t+u)+2*m2*(2*m4*m32+(2*m42-2*m_X2+u)*m3+m4*(2*m_X2-s-t)))*m12-2*((4*m4*m32+(2*m42+2*m_X2+u)*m3+m4*(2*m42-2*m_X2-2*s-2*t+u))*m22-(m32+2*m4*m3+m42-s)*(4*m_X2-u)*m2+m32*m4*(2*m42-2*m_X2-2*s-2*t+u)+m3*(m44+(2*m_X2-s-t)*m42-2*m_X2*t-s*u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+2*s*t+2*m_X2*u))*m1-m42*s2+4*m_X2*s2-m42*t2+m42*u2+m44*s+3*m32*m42*s-4*m32*m_X2*s-4*m42*m_X2*s-8*m3*m4*m_X2*s+m44*t+m32*m42*t-2*m42*s*t+m22*(-4*(m42-m_X2)*m32+m4*(8*m_X2-2*u)*m3-4*m_X2*s+m42*(4*m_X2+s+3*t-u))-m44*u-m32*m42*u+2*m3*m4*s*u+2*m2*(m45-(s+t+u)*m43+2*m_X2*t*m4+m32*(2*m42-2*m_X2-u)*m4+s*u*m4+m3*(m42+2*m_X2)*(m42-u))) * u_propX*(u-m_X2)*(-s_proph)*(s-m_h2)

        thsX = -(1/(m_X4))*8*m_d2*(2*(m2-m4)*m15+(4*m22+2*m3*m2-2*m3*m4+s-t-u)*m14+2*(2*m23+2*(m32+m42-t-u)*m2+m4*(-2*m32+s+t+u))*m13+(4*m24+(8*m32+4*m4*m3+8*m42-2*(2*m_X2+s+3*(t+u)))*m22+2*(m4*(s-4*m_X2)+m3*(2*m42+2*m_X2-t-u))*m2-4*m42*m_X2-s2+t2+u2+m42*s+2*m3*m4*(s-2*m_X2)-m42*t+4*m_X2*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m12+2*(m25+m4*m24+2*(m32+m42-t-u)*m23+(2*m4*m32+(s-4*m_X2)*m3+m4*(2*m_X2-t-u))*m22+((4*m42-2*m_X2+s-2*t-2*u)*m32+2*m4*(s-4*m_X2)*m3-s2+t2+u2+2*m_X2*s+2*t*u+m42*(-2*m_X2+s-2*(t+u)))*m2+m32*m4*(2*m_X2+s)-m3*(4*m_X2-s)*(m42-t)-m4*(2*u*m_X2+s*t))*m1-2*m25*m3+4*m_X2*((t-m42)*m32+m4*s*m3+(m42-t)*t)-m24*(2*m3*m4-s+t+u)+2*m23*m3*(-2*m42+s+t+u)+m22*((4*m42-4*m_X2+s-t-3*u)*m32+2*m4*(s-2*m_X2)*m3-s2+t2+u2+4*m_X2*t+m42*(s-3*t-u)+2*t*u)-2*m2*(m4*(4*m_X2-s)*m32+(-((2*m_X2+s)*m42)+s*t+2*m_X2*u)*m3+m4*(s-4*m_X2)*t)) * s_propX*(s-m_X2)*(-t_proph)*(t-m_h2)

        uhsX = -(1/(m_X4))*8*m_d2*(2*(m2-m3)*m15+(4*m22+2*m4*m2-2*m3*m4+s-t-u)*m14+2*(2*m23+2*(m32+m42-t-u)*m2+m3*(-2*m42+s+t+u))*m13+(4*m24+(8*m32+4*m4*m3+8*m42-2*(2*m_X2+s+3*(t+u)))*m22+2*(2*m4*m32+(s-4*m_X2)*m3+m4*(2*m_X2-t-u))*m2-s2+t2+u2+m42*s+2*m3*m4*(s-2*m_X2)-m42*t+m32*(4*m42-4*m_X2+s-3*t-u)-3*m42*u+4*m_X2*u+2*t*u)*m12+2*(m25+m3*m24+2*(m32+m42-t-u)*m23+(m4*(s-4*m_X2)+m3*(2*m42+2*m_X2-t-u))*m22+((4*m42-2*m_X2+s-2*t-2*u)*m32+2*m4*(s-4*m_X2)*m3-s2+t2+u2+2*m_X2*s+2*t*u+m42*(-2*m_X2+s-2*(t+u)))*m2+2*m3*m42*m_X2-4*m32*m4*m_X2+m3*m42*s+m32*m4*s-2*m3*m_X2*t+4*m4*m_X2*u-m3*s*u-m4*s*u)*m1-2*m25*m4-m24*(2*m3*m4-s+t+u)+2*m23*m4*(-2*m32+s+t+u)+4*m_X2*((u-m42)*m32+m4*s*m3+(m42-u)*u)+m22*((4*m42+s-t-3*u)*m32+2*m4*(s-2*m_X2)*m3-s2+t2+u2+4*m_X2*u+2*t*u-m42*(4*m_X2-s+3*t+u))+2*m2*(m4*(2*m_X2+s)*m32-(4*m_X2-s)*(m42-u)*m3-m4*(2*t*m_X2+s*u))) * s_propX*(s-m_X2)*(-u_proph)*(u-m_h2)

    else:
        # BW-propagators
        # D_BW(X) = X_prop*((X-m^2)-imG)
        # Re(D_BW(X)* x D_BW(Y)) = X_prop*Y_prop*((X-mx^2)*(Y-my^2)+mxG*myG)
        # sproph = s_proph*((s-m_h2)-1j*m_Gamma_h)
        # tproph = t_proph*((t-m_h2)-1j*m_Gamma_h)
        # uproph = u_proph*((u-m_h2)-1j*m_Gamma_h)
        # spropX = s_propX*((s-m_X2)-1j*m_Gamma_X)
        # tpropX = t_propX*((t-m_X2)-1j*m_Gamma_X)
        # upropX = u_propX*((u-m_X2)-1j*m_Gamma_X)
        # Squared BW propagators 
        # |D_BW|^2 = X_prop
        # sproph2 = s_proph
        # tproph2 = t_proph
        # uproph2 = u_proph
        # spropX2 = s_propX
        # tpropX2 = t_propX
        # upropX2 = u_propX

        ssh = (64*m_d4*(m12+2*m2*m1+m22-s)*(m32+2*m4*m3+m42-s))/m_X4 * s_proph

        sth = (32*m_d4*((-2*m42-2*m3*m4+2*m2*(m3-m4)+s+t-u)*m12-2*((m3-m4)*m22+(m32+2*m4*m3+m42-s)*m2+m3*m42-m32*m4-m3*t+m4*u)*m1-s2-t2+u2+m32*s+m42*s+2*m3*m4*s+m32*t+m42*t+m22*(-2*m32-2*m4*m3+s+t-u)-m32*u-m42*u-2*m2*(m4*m32-m42*m3+u*m3-m4*t)))/m_X4 * s_proph*t_proph*((s-m_h2)*(t-m_h2)+m_Gamma_h2)

        suh = (32*m_d4*((-2*m32-2*m2*m3-2*m4*m3+2*m2*m4+s-t+u)*m12+2*((m3-m4)*m22-(m32+2*m4*m3+m42-s)*m2+m3*m42-m32*m4-m3*t+m4*u)*m1-s2+t2-u2+m32*s+m42*s+2*m3*m4*s-m32*t-m42*t+m32*u+m42*u+m22*(-2*m42-2*m3*m4+s-t+u)+2*m2*(m4*m32-m42*m3+u*m3-m4*t)))/m_X4 * s_proph*u_proph*((s-m_h2)*(u-m_h2)+m_Gamma_h2)

        ssX = (1/(m_X4))*4*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+2*m2*(s-2*m_X2)*m15+(2*m24+(2*m32+2*m42-8*m_X2+3*s-2*t-2*u)*m22+s2-t2-u2+m3*m4*(4*m_X2-2*s)+m42*s-2*m_X2*s+2*m42*t+2*m_X2*t-2*s*t+2*m42*u+2*m_X2*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))*m14-4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+((2*m32+2*m42-8*m_X2+3*s-2*t-2*u)*m24+2*(2*m_X4+2*s*m_X2+6*t*m_X2+6*u*m_X2-s2+t2+u2+2*m3*m4*(s-2*m_X2)-2*s*t+m32*(4*m42-8*m_X2+3*s-2*t-2*u)-2*s*u+2*t*u-m42*(8*m_X2-3*s+2*(t+u)))*m22+2*m42*m_X4-s3+m42*s2+2*m_X2*s2-2*m_X2*t2+s*t2-2*m_X2*u2+s*u2-2*m42*m_X2*s+2*m3*m4*(2*m_X4-2*s*m_X2+s2)-2*m_X4*t+2*m42*m_X2*t-2*m42*s*t-2*m_X4*u+6*m42*m_X2*u-2*m42*s*u-4*m_X2*t*u+2*s*t*u+m32*(2*m_X4+(-2*s+6*t+2*u)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))*m12+2*m2*((s-2*m_X2)*m24-2*(2*m_X2-s)*(m32+m42-t-u)*m22+2*m42*m_X4-s3+m42*s2+2*m_X2*s2-2*m_X2*t2+s*t2-2*m_X2*u2+s*u2-2*m_X4*s-2*m42*m_X2*s+2*m3*m4*(4*m_X4-2*s*m_X2+s2)+4*m42*m_X2*t-2*m42*s*t+4*m42*m_X2*u-2*m42*s*u-4*m_X2*t*u+2*s*t*u+m32*(2*m_X4+(4*(t+u)-2*s)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32-2*m4*s*m3+t2+u2-m42*(t+u))+m24*((-4*m42+s+2*(t+u))*m32+m4*(4*m_X2-2*s)*m3+s2-t2-u2-2*m_X2*s+2*m_X2*t-2*s*t+2*m_X2*u-2*s*u-2*t*u+m42*(s+2*(t+u)))+m22*(-2*t*m_X4-2*u*m_X4+2*s2*m_X2-2*t2*m_X2-2*u2*m_X2-4*t*u*m_X2-s3+s*t2+s*u2+2*m3*m4*(2*m_X4-2*s*m_X2+s2)+2*s*t*u+m42*(2*m_X4+(-2*s+6*t+2*u)*m_X2+s*(s-2*(t+u)))+m32*(2*m_X4+(-2*s+2*t+6*u)*m_X2+m42*(4*s-8*m_X2)+s*(s-2*(t+u))))) * s_propX

        stX = (1/(m_X4))*2*(2*m18+(4*m22+4*m32+4*m42-3*s-3*t-5*u)*m16+2*(2*m23+2*(m3-m4)*m22+(2*m32-s-u)*m2+2*m33-2*m32*m4+m4*(4*m_X2+u)-m3*(t+u))*m15+(2*m24+4*(m3-m4)*m23+(4*m32-4*m4*m3+4*m42-2*(4*m_X2+s+t+3*u))*m22+2*(2*m33-2*m4*m32-(4*m_X2+u)*m3+m4*(t+u))*m2+2*m34+4*u2-4*m33*m4-3*m42*s-3*m42*t+4*s*t-5*m42*u+4*m_X2*u+4*s*u+4*t*u+2*m3*m4*(s+u)+m32*(4*m42-2*(4*m_X2+s+t+3*u)))*m14+(4*m25+4*(m3-m4)*m24+(8*m32+8*m42-4*(2*m_X2+s+t+2*u))*m23+2*(4*m33-4*m4*m32+4*m42*m3-2*(2*m_X2+s+t+2*u)*m3+m4*(4*m_X2+s+3*(t+u)))*m22+(4*m34+(8*m42-4*(2*m_X2+s+t+2*u))*m32+s2-t2+3*u2-4*m_X2*s+4*m_X2*t+4*s*t+4*m_X2*u+4*s*u+2*t*u-4*m42*(s+u))*m2+4*m35-4*m34*m4-2*m4*(4*m_X2+u)*(s+t+u)+2*m32*m4*(4*m_X2+3*s+t+3*u)+m3*(4*(s-t+u)*m_X2-s2-(4*m42-t-3*u)*(t+u)+2*s*(2*t+u))+m33*(8*m42-4*(2*m_X2+s+t+2*u)))*m13+(4*(m3-m4)*m25+(-8*m_X2-4*m3*m4+s+t-u)*m24+2*(4*m33-4*m4*m32+4*m42*m3-2*(s+t+2*u)*m3+m4*s+3*m4*(t+u))*m23+(-8*m4*m33+(4*m42-32*m_X2+3*(s+t-u))*m32+2*m4*(3*(s+t+u)-8*m_X2)*m3+m42*(-24*m_X2+s+t-u)+2*(4*m_X4+(5*s+5*t+7*u)*m_X2-s2-t2+u2))*m22+(4*m35-4*m4*m34+(8*m42-4*(s+t+2*u))*m33+2*m4*(3*(s+t+u)-8*m_X2)*m32-(-8*m_X4-8*u*m_X2+s2+t2-3*u2-4*s*t-2*s*u-2*t*u+4*m42*(4*m_X2+u))*m3+2*m4*(4*m_X4-2*(s-u)*m_X2-t2-2*t*u-u*(s+u)))*m2+s3+t3-u3-m42*s2-m42*t2-s*t2+m42*u2-4*m_X2*u2-s*u2-t*u2-4*m35*m4-s2*t+4*m42*s*t+m34*(-8*m_X2+s+t-u)-8*m_X4*u+12*m42*m_X2*u+s2*u+t2*u-4*m_X2*s*u-4*m_X2*t*u-2*s*t*u+2*m33*m4*(3*s+t+3*u)+2*m3*m4*(4*m_X4-2*(t-u)*m_X2-s2-2*s*u-u*(t+u))+m32*((-24*m_X2+s+t-u)*m42+2*(4*m_X4+(5*s+5*t+7*u)*m_X2-s2-t2+u2)))*m12-(2*(2*m32+4*m4*m3+2*m_X2-s+u)*m25+2*(2*m33+2*m4*m32+(2*m_X2-s-t+u)*m3+m4*(2*m_X2-t))*m24+(4*m34+16*m4*m33+2*(4*m_X2-3*s-3*t+u)*m32-8*m4*(s+t+u)*m3+3*s2+t2-3*u2-8*m_X2*t-8*m_X2*u-2*t*u+2*m42*(4*m_X2-s+t+u))*m23+(4*m35+4*m4*m34+2*(4*m_X2-3*s-3*t+u)*m33+2*m4*(8*m_X2-3*s-3*t-u)*m32+(-8*m_X4-4*(s+3*u)*m_X2+2*m42*(12*m_X2-s-t+u)+3*(s2+t2-u2))*m3+2*m4*(-4*m_X4+2*(s+t-u)*m_X2+t*(t+u)))*m22+(8*m4*m35+2*(2*m_X2-s-t+u)*m34-8*m4*(s+t+u)*m33+(-8*m_X4-4*(t+3*u)*m_X2+2*m42*(12*m_X2-s-t+u)+3*(s2+t2-u2))*m32+2*m4*(-16*m_X4+4*(s+t)*m_X2+s2+(t+u)**2+2*s*u)*m3-s3-t3+u3-4*m_X2*s2+4*m_X2*t2+s*t2+4*m_X2*u2+s*u2+t*u2+8*m_X4*s+s2*t-4*m_X2*s*t-s2*u-t2*u+4*m_X2*t*u+2*s*t*u+m42*(-8*m_X4+4*(s-2*u)*m_X2+s2+t2-u2-4*s*t))*m2+m34*m4*(4*m_X2-2*s)+2*m35*(2*m_X2-t+u)+2*m32*m4*(-4*m_X4+2*(s+t-u)*m_X2+s*(s+u))+m33*(2*(4*m_X2+s-t+u)*m42+s2+3*t2-3*u2-2*s*u-8*m_X2*(s+u))+4*m4*m_X2*((2*m_X2-t)*u-s*(2*t+u))+m3*(8*t*m_X4+4*(s2-t*s+u*s-t2+u2)*m_X2-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(-8*m_X4+4*(t-2*u)*m_X2+s2+t2-u2-4*s*t)))*m1+m25*(-4*m33-4*m4*m32+2*(2*m_X2+s-u)*m3+4*m4*m_X2)+m24*(-2*m34-4*m4*m33+(s+t-u)*m32+2*m4*(2*m_X2+t)*m3+2*m_X2*(-s+t+u))+2*m_X2*(2*m4*m35+(s-t+u)*m34-2*m4*(s+t+u)*m33+((4*m_X2+3*s-t+u)*m42-s2+t2-u2-4*m_X2*u-2*s*u)*m32+2*m4*(t*(s+u)-2*m_X2*s)*m3+4*m_X2*u*(u-m42))-m23*(4*m35+4*m4*m34+(-6*s-6*t+2*u)*m33-2*m4*(4*m_X2+3*s+t+u)*m32+(-2*(4*m_X2+s-t-u)*m42+3*s2+t2-3*u2-2*t*u+4*m_X2*(s+t+u))*m3+4*m4*m_X2*(s+t+u))+m22*(-4*m4*m35+(s+t-u)*m34+2*m4*(4*m_X2+s+3*t+u)*m33+((-16*m_X2+s+t-u)*m42-s2-t2+u2+12*m_X2*u)*m32+2*m4*(4*m_X4-2*(s+2*t)*m_X2-t*(t+u))*m3+2*m_X2*((4*m_X2-s+3*t+u)*m42+s2-t2-u2-4*m_X2*u-2*t*u))+m2*(2*(2*m_X2+t-u)*m35+2*m4*(2*m_X2+s)*m34+(2*(4*m_X2-s+t-u)*m42-s2-3*t2+3*u2+2*s*u-4*m_X2*(s+t+u))*m33+2*m4*(4*m_X4-2*(2*s+t)*m_X2-s*(s+u))*m32+(-8*u*m_X4+4*t*u*m_X2+4*s*(2*t+u)*m_X2+s3+t3-u3-t*u2-s*(t+u)**2+t2*u+s2*(u-t)+m42*(8*m_X4-8*(s+t)*m_X2-s2-t2+u2+4*s*t))*m3+4*m4*m_X2*(s*(t+u)-2*m_X2*t))) * s_propX*t_propX*((s-m_X2)*(t-m_X2)+m_Gamma_X2)

        suX = (1/(m_X4))*2*(2*m18+(4*m22+4*m32+4*m42-3*s-5*t-3*u)*m16+2*(2*m23-2*(m3-m4)*m22+(2*m42-s-t)*m2+m3*(-2*m42+4*m_X2+t)+m4*(2*m42-t-u))*m15+(2*m24-4*(m3-m4)*m23+(4*m32-4*m4*m3+4*m42-2*(4*m_X2+s+3*t+u))*m22+(4*m43-2*(4*m_X2+t)*m4+2*m3*(-2*m42+t+u))*m2+2*m44+4*m_X2*t+2*m3*m4*(-2*m42+s+t)+m32*(4*m42-3*s-5*t-3*u)+4*(s+t)*(t+u)-2*m42*(4*m_X2+s+3*t+u))*m14+(4*m25-4*(m3-m4)*m24+(8*m32+8*m42-4*(2*m_X2+s+2*t+u))*m23+2*(4*m4*m32+(-4*m42+4*m_X2+s+3*t+3*u)*m3+2*m4*(2*m42-2*m_X2-s-2*t-u))*m22+(4*m44-4*(2*m_X2+s+2*t+u)*m42+s2+3*t2-u2-4*m_X2*s+4*m_X2*t+4*s*t+m32*(8*m42-4*(s+t))+4*m_X2*u+4*s*u+2*t*u)*m2+m32*(8*m43-4*m4*(t+u))+m4*(4*m44-4*(2*m_X2+s+2*t+u)*m42-s2+3*t2+u2+2*s*t+4*m_X2*(s+t-u)+4*s*u+4*t*u)-2*m3*(2*m44-(4*m_X2+3*s+3*t+u)*m42+(4*m_X2+t)*(s+t+u)))*m13+(-4*(m3-m4)*m25+(-8*m_X2-4*m3*m4+s-t+u)*m24+2*(4*m4*m32+(-4*m42+s+3*(t+u))*m3+2*m4*(2*m42-s-2*t-u))*m23+((4*m42-24*m_X2+s-t+u)*m32+2*m4*(-4*m42-8*m_X2+3*(s+t+u))*m3+m42*(3*(s-t+u)-32*m_X2)+2*(4*m_X4+(5*s+7*t+5*u)*m_X2-s2+t2-u2))*m22+(4*m4*(2*m42-4*m_X2-t)*m32-2*(2*m44+(8*m_X2-3*(s+t+u))*m42-4*m_X4+t2+u2+2*m_X2*(s-t)+s*t+2*t*u)*m3+m4*(4*m44-4*(s+2*t+u)*m42+8*m_X4-s2+3*t2-u2+8*m_X2*t+2*s*t+4*s*u+2*t*u))*m2+8*m42*m_X4+s3-t3+u3-8*m44*m_X2-2*m42*s2+2*m42*t2-4*m_X2*t2-s*t2-2*m42*u2-s*u2+t*u2+m44*s+10*m42*m_X2*s-m44*t-8*m_X4*t+14*m42*m_X2*t+s2*t-4*m_X2*s*t+m44*u+10*m42*m_X2*u-s2*u-t2*u-4*m_X2*t*u-2*s*t*u+m32*((-24*m_X2+s-t+u)*m42-s2+t2-u2+12*m_X2*t+4*s*u)-2*m3*m4*(2*m44-(3*s+3*t+u)*m42-4*m_X4+s2+t2+2*s*t-2*m_X2*(t-u)+t*u))*m12-(2*(2*m42+4*m3*m4+2*m_X2-s+t)*m25+2*(m3*(2*m42+2*m_X2-u)+m4*(2*m42+2*m_X2-s+t-u))*m24+(4*m44+2*(4*m_X2-3*s+t-3*u)*m42+8*m3*(2*m42-s-t-u)*m4+3*s2-3*t2+u2-8*m_X2*t-8*m_X2*u-2*t*u+2*m32*(4*m_X2-s+t+u))*m23+(2*m4*(12*m_X2-s+t-u)*m32+2*(2*m44+(8*m_X2-3*s-t-3*u)*m42-4*m_X4+2*m_X2*(s-t+u)+u*(t+u))*m3+m4*(4*m44+2*(4*m_X2-3*s+t-3*u)*m42-8*m_X4-4*m_X2*(s+3*t)+3*(s2-t2+u2)))*m22+(2*(2*m_X2-s+t-u)*m44-(8*m_X4+4*(3*t+u)*m_X2-3*(s2-t2+u2))*m42+2*m3*(4*m44-4*(s+t+u)*m42-16*m_X4+s2+t2+u2+2*s*t+2*t*u+4*m_X2*(s+u))*m4-s3+t3-u3-4*m_X2*s2+4*m_X2*t2+s*t2+4*m_X2*u2+s*u2-t*u2+8*m_X4*s-s2*t+s2*u+t2*u-4*m_X2*s*u+4*m_X2*t*u+2*s*t*u+m32*(-8*m_X4+4*(s-2*t)*m_X2+s2-t2+u2+2*m42*(12*m_X2-s+t-u)-4*s*u))*m2+m32*m4*(-8*m_X4+(4*u-8*t)*m_X2+s2-t2+u2+2*m42*(4*m_X2+s+t-u)-4*s*u)+m4*(2*(2*m_X2+t-u)*m44+(-8*(s+t)*m_X2+s2-3*t2+3*u2-2*s*t)*m42-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+8*m_X4*u+s2*u+t2*u+2*s*t*u+4*m_X2*(s2+t*s-u*s+t2-u2))+2*m3*((2*m_X2-s)*m44+(-4*m_X4+2*(s-t+u)*m_X2+s*(s+t))*m42+2*m_X2*(2*t*m_X2-t*u-s*(t+2*u))))*m1+m25*(2*m4*(-2*m42+2*m_X2+s-t)-4*m3*(m42-m_X2))+m24*(-2*m44+(s-t+u)*m42+2*m3*(-2*m42+2*m_X2+u)*m4+2*m_X2*(-s+t+u))+2*m_X2*((s+t-u)*m44-(4*t*m_X2+s2+t2-u2+2*s*t)*m42+2*m3*(m44-(s+t+u)*m42-2*m_X2*s+(s+t)*u)*m4+4*m_X2*t2+m32*(m42*(4*m_X2+3*s+t-u)-4*m_X2*t))+m23*(2*m4*(4*m_X2+s-t-u)*m32+(-4*m44+2*(4*m_X2+3*s+t+u)*m42-4*m_X2*(s+t+u))*m3-m4*(4*m44+(-6*s+2*t-6*u)*m42+3*s2-3*t2+u2-2*t*u+4*m_X2*(s+t+u)))+m22*((s-t+u)*m44-(-12*t*m_X2+s2-t2+u2)*m42-2*m3*(2*m44-(4*m_X2+s+t+3*u)*m42-4*m_X4+u*(t+u)+2*m_X2*(s+2*u))*m4-2*m_X2*(4*t*m_X2-s2+(t+u)**2)+m32*((-16*m_X2+s-t+u)*m42+2*m_X2*(4*m_X2-s+t+3*u)))+m2*(m4*(8*m_X4-8*(s+u)*m_X2-s2+t2-u2+4*s*u+2*m42*(4*m_X2-s-t+u))*m32+2*((2*m_X2+s)*m44+(4*m_X4-2*(2*s+u)*m_X2-s*(s+t))*m42-4*m_X4*u+2*m_X2*s*(t+u))*m3+m4*(2*(2*m_X2-t+u)*m44-(4*(s+t+u)*m_X2+s2-3*t2+3*u2-2*s*t)*m42+s3-t3+u3+t*u2-s*(t+u)**2-8*m_X4*t+s2*(t-u)-t2*u+4*m_X2*t*u+4*m_X2*s*(t+2*u)))) * s_propX*u_propX*((s-m_X2)*(u-m_X2)+m_Gamma_X2)

        shtX = -(1/(m_X4))*8*m_d2*(2*(m3-m4)*m15+(4*m32+2*m2*(m3-m4)-s+t-u)*m14+2*(2*m33+2*(m42-s-u)*m3+2*m22*(m3-m4)+m4*(s+t+u))*m13+(4*m34+(8*m42-2*(2*m_X2+3*s+t+3*u))*m32+2*m4*(t-4*m_X2)*m3-4*m42*m_X2+s2-t2+u2-m42*s+4*m_X2*s+m42*t+2*m2*(2*m4*m32+(2*m42+2*m_X2-s-u)*m3+m4*(t-2*m_X2))+m22*(8*m32+4*m42-3*s+t-u)-3*m42*u+2*s*u)*m12+2*(m35+m4*m34+2*(m42-s-u)*m33+m4*(2*m_X2-s-u)*m32+((-2*m_X2-2*s+t-2*u)*m42+s2-t2+u2+2*m_X2*t+2*s*u)*m3-m2*(m32+2*m4*m3+m42-s)*(4*m_X2-t)+m22*(2*m33+2*m4*m32+(4*m42-2*m_X2-2*s+t-2*u)*m3+m4*(2*m_X2+t))-m4*(2*u*m_X2+s*t))*m1+m32*s2-4*m_X2*s2-m32*t2+m32*u2-m34*s-3*m32*m42*s+4*m32*m_X2*s+4*m42*m_X2*s+8*m3*m4*m_X2*s+m34*t+m32*m42*t-2*m3*m4*s*t+m22*((4*m42-4*m_X2-s+t-3*u)*m32+2*m4*(t-4*m_X2)*m3+4*m_X2*(s-m42))-m34*u-m32*m42*u+2*m32*s*u-2*m2*(m35+m4*m34+(2*m42-s-t-u)*m33+m4*(2*m_X2-t)*m32+(-((2*m_X2+t)*m42)+s*t+2*m_X2*u)*m3-2*m4*m_X2*t)) * t_propX*(-s_proph)*((t-m_X2)*(s-m_h2)+m_Gamma_X*m_Gamma_h)

        shuX = (1/(m_X4))*8*m_d2*(2*(m3-m4)*m15+(-4*m42+2*m2*(m3-m4)+s+t-u)*m14+(4*m22*(m3-m4)-2*(2*m4*m32+(s+t+u)*m3+2*m4*(m42-s-t)))*m13-(4*m44-4*m_X2*m42-6*s*m42-6*t*m42-2*u*m42+2*m3*(u-4*m_X2)*m4+s2+t2-u2+4*m_X2*s+2*s*t+m32*(8*m42-4*m_X2-s-3*t+u)+m22*(4*m32+8*m42-3*s-t+u)+2*m2*(2*m4*m32+(2*m42-2*m_X2+u)*m3+m4*(2*m_X2-s-t)))*m12-2*((4*m4*m32+(2*m42+2*m_X2+u)*m3+m4*(2*m42-2*m_X2-2*s-2*t+u))*m22-(m32+2*m4*m3+m42-s)*(4*m_X2-u)*m2+m32*m4*(2*m42-2*m_X2-2*s-2*t+u)+m3*(m44+(2*m_X2-s-t)*m42-2*m_X2*t-s*u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+2*s*t+2*m_X2*u))*m1-m42*s2+4*m_X2*s2-m42*t2+m42*u2+m44*s+3*m32*m42*s-4*m32*m_X2*s-4*m42*m_X2*s-8*m3*m4*m_X2*s+m44*t+m32*m42*t-2*m42*s*t+m22*(-4*(m42-m_X2)*m32+m4*(8*m_X2-2*u)*m3-4*m_X2*s+m42*(4*m_X2+s+3*t-u))-m44*u-m32*m42*u+2*m3*m4*s*u+2*m2*(m45-(s+t+u)*m43+2*m_X2*t*m4+m32*(2*m42-2*m_X2-u)*m4+s*u*m4+m3*(m42+2*m_X2)*(m42-u))) * u_propX*(-s_proph)*((u-m_X2)*(s-m_h2)+m_Gamma_X*m_Gamma_h)

        thsX = -(1/(m_X4))*8*m_d2*(2*(m2-m4)*m15+(4*m22+2*m3*m2-2*m3*m4+s-t-u)*m14+2*(2*m23+2*(m32+m42-t-u)*m2+m4*(-2*m32+s+t+u))*m13+(4*m24+(8*m32+4*m4*m3+8*m42-2*(2*m_X2+s+3*(t+u)))*m22+2*(m4*(s-4*m_X2)+m3*(2*m42+2*m_X2-t-u))*m2-4*m42*m_X2-s2+t2+u2+m42*s+2*m3*m4*(s-2*m_X2)-m42*t+4*m_X2*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m12+2*(m25+m4*m24+2*(m32+m42-t-u)*m23+(2*m4*m32+(s-4*m_X2)*m3+m4*(2*m_X2-t-u))*m22+((4*m42-2*m_X2+s-2*t-2*u)*m32+2*m4*(s-4*m_X2)*m3-s2+t2+u2+2*m_X2*s+2*t*u+m42*(-2*m_X2+s-2*(t+u)))*m2+m32*m4*(2*m_X2+s)-m3*(4*m_X2-s)*(m42-t)-m4*(2*u*m_X2+s*t))*m1-2*m25*m3+4*m_X2*((t-m42)*m32+m4*s*m3+(m42-t)*t)-m24*(2*m3*m4-s+t+u)+2*m23*m3*(-2*m42+s+t+u)+m22*((4*m42-4*m_X2+s-t-3*u)*m32+2*m4*(s-2*m_X2)*m3-s2+t2+u2+4*m_X2*t+m42*(s-3*t-u)+2*t*u)-2*m2*(m4*(4*m_X2-s)*m32+(-((2*m_X2+s)*m42)+s*t+2*m_X2*u)*m3+m4*(s-4*m_X2)*t)) * s_propX*(-t_proph)*((s-m_X2)*(t-m_h2)+m_Gamma_X*m_Gamma_h)

        uhsX = -(1/(m_X4))*8*m_d2*(2*(m2-m3)*m15+(4*m22+2*m4*m2-2*m3*m4+s-t-u)*m14+2*(2*m23+2*(m32+m42-t-u)*m2+m3*(-2*m42+s+t+u))*m13+(4*m24+(8*m32+4*m4*m3+8*m42-2*(2*m_X2+s+3*(t+u)))*m22+2*(2*m4*m32+(s-4*m_X2)*m3+m4*(2*m_X2-t-u))*m2-s2+t2+u2+m42*s+2*m3*m4*(s-2*m_X2)-m42*t+m32*(4*m42-4*m_X2+s-3*t-u)-3*m42*u+4*m_X2*u+2*t*u)*m12+2*(m25+m3*m24+2*(m32+m42-t-u)*m23+(m4*(s-4*m_X2)+m3*(2*m42+2*m_X2-t-u))*m22+((4*m42-2*m_X2+s-2*t-2*u)*m32+2*m4*(s-4*m_X2)*m3-s2+t2+u2+2*m_X2*s+2*t*u+m42*(-2*m_X2+s-2*(t+u)))*m2+2*m3*m42*m_X2-4*m32*m4*m_X2+m3*m42*s+m32*m4*s-2*m3*m_X2*t+4*m4*m_X2*u-m3*s*u-m4*s*u)*m1-2*m25*m4-m24*(2*m3*m4-s+t+u)+2*m23*m4*(-2*m32+s+t+u)+4*m_X2*((u-m42)*m32+m4*s*m3+(m42-u)*u)+m22*((4*m42+s-t-3*u)*m32+2*m4*(s-2*m_X2)*m3-s2+t2+u2+4*m_X2*u+2*t*u-m42*(4*m_X2-s+3*t+u))+2*m2*(m4*(2*m_X2+s)*m32-(4*m_X2-s)*(m42-u)*m3-m4*(2*t*m_X2+s*u))) * s_propX*(-u_proph)*((s-m_X2)*(u-m_h2)+m_Gamma_X*m_Gamma_h)

    tth = (64*m_d4*(m12+2*m3*m1+m32-t)*(m22+2*m4*m2+m42-t))/m_X4 * t_proph
        
    uuh = (64*m_d4*(m22+2*m3*m2+m32-u)*(m12+2*m4*m1+m42-u))/m_X4 * u_proph

    tuh = -((32*m_d4*((2*m22+2*(m3+m4)*m2-2*m3*m4+s-t-u)*m12+2*((m3+m4)*m22-(m32-2*m4*m3+m42-s)*m2+m3*m42+m32*m4-m3*t-m4*u)*m1+2*m32*m42-s2+t2+u2+m32*s+m42*s+2*m3*m4*s-m32*t-m42*t-m32*u-m42*u-m22*(2*m3*m4-s+t+u)+2*m2*(m4*m32+m42*m3-u*m3-m4*t)))/m_X4) * t_proph*u_proph*((t-m_h2)*(u-m_h2)+m_Gamma_h2)

    ttX = (1/(m_X4))*4*(-m18+(-2*m22-2*m42+2*s+t+2*u)*m16+2*m3*(t-2*m_X2)*m15+(2*m34+(2*m42-8*m_X2-2*s+3*t-2*u)*m32-s2+t2-u2+2*m42*s+2*m_X2*s+m2*m4*(4*m_X2-2*t)+m42*t-2*m_X2*t-2*s*t+2*m42*u+2*m_X2*u-2*s*u-2*t*u+m22*(2*m32-4*m42+2*s+t+2*u))*m14-4*m3*(2*m_X2-t)*(m22+m32+m42-s-u)*m13+((2*m42-8*m_X2-2*s+3*t-2*u)*m34-2*(-2*m_X4-2*(3*s+t+3*u)*m_X2-s2+t2-u2+2*s*t-2*s*u+2*t*u+m42*(8*m_X2+2*s-3*t+2*u))*m32+2*m42*m_X4-t3-2*m_X2*s2+m42*t2+2*m_X2*t2-2*m_X2*u2+t*u2-2*m_X4*s+2*m42*m_X2*s-2*m42*m_X2*t+s2*t-2*m42*s*t+2*m2*m4*(2*m_X4-2*t*m_X2+t2+m32*(2*t-4*m_X2))-2*m_X4*u+6*m42*m_X2*u-4*m_X2*s*u-2*m42*t*u+2*s*t*u+m22*(2*m34+2*(4*m42-8*m_X2-2*s+3*t-2*u)*m32+2*m_X4+t2+6*m_X2*s-2*m_X2*t-2*s*t+m42*(4*t-8*m_X2)+2*m_X2*u-2*t*u))*m12+2*m3*((t-2*m_X2)*m34-2*(2*m_X2-t)*(m42-s-u)*m32+2*m42*m_X4-t3-2*m_X2*s2+m42*t2+2*m_X2*t2-2*m_X2*u2+t*u2+4*m42*m_X2*s-2*m_X4*t-2*m42*m_X2*t+s2*t-2*m42*s*t+2*m2*m4*(4*m_X4-2*t*m_X2+t2)+4*m42*m_X2*u-4*m_X2*s*u-2*m42*t*u+2*s*t*u+m22*(2*m_X4+4*s*m_X2-2*t*m_X2+4*u*m_X2+t2-2*s*t+m32*(2*t-4*m_X2)+m42*(4*t-8*m_X2)-2*t*u))*m1-m38+2*m32*m42*m_X4-m32*t3-2*m36*m42-m34*s2+2*m_X4*s2-2*m32*m_X2*s2+m34*t2+m32*m42*t2+2*m32*m_X2*t2-m34*u2+2*m_X4*u2-2*m32*m_X2*u2+m32*t*u2+2*m36*s-2*m32*m_X4*s-2*m42*m_X4*s+2*m34*m42*s+2*m34*m_X2*s+6*m32*m42*m_X2*s+m36*t+m34*m42*t-2*m34*m_X2*t-2*m32*m42*m_X2*t+m32*s2*t-2*m34*s*t-2*m32*m42*s*t+2*m2*m4*((2*m_X2-t)*m34+(2*m_X4-2*t*m_X2+t2)*m32-2*m_X4*t)+2*m36*u-2*m32*m_X4*u-2*m42*m_X4*u+2*m34*m42*u+2*m34*m_X2*u+2*m32*m42*m_X2*u-2*m34*s*u-4*m32*m_X2*s*u-2*m34*t*u-2*m32*m42*t*u+2*m32*s*t*u+m22*(-2*m36+(-4*m42+2*s+t+2*u)*m34+(2*m_X4+2*(s-t+3*u)*m_X2+m42*(4*t-8*m_X2)+t*(-2*s+t-2*u))*m32+2*m_X4*(2*m42-s-u))) * t_propX

    uuX = (1/(m_X4))*4*(-m18+(-2*m22-2*m32+2*s+2*t+u)*m16+2*m4*(u-2*m_X2)*m15+(2*m44-8*m_X2*m42-2*s*m42-2*t*m42+3*u*m42-s2-t2+u2+2*m_X2*s+2*m_X2*t-2*s*t+m2*m3*(4*m_X2-2*u)-2*m_X2*u-2*s*u-2*t*u+m32*(2*m42+2*s+2*t+u)+m22*(-4*m32+2*m42+2*s+2*t+u))*m14-4*m4*(m22+m32+m42-s-t)*(2*m_X2-u)*m13+(-8*m_X2*m44-2*s*m44-2*t*m44+3*u*m44+4*m_X4*m42+2*s2*m42+2*t2*m42-2*u2*m42+12*m_X2*s*m42+12*m_X2*t*m42+4*s*t*m42+4*m_X2*u*m42-4*s*u*m42-4*t*u*m42-u3-2*m_X2*s2-2*m_X2*t2+2*m_X2*u2-2*m_X4*s-2*m_X4*t-4*m_X2*s*t+s2*u+t2*u+2*s*t*u+m22*(2*m44-2*(8*m_X2+2*s+2*t-3*u)*m42+2*m_X4+u2+6*m_X2*s+2*m_X2*t-2*m_X2*u-2*s*u-2*t*u+4*m32*(2*m42-2*m_X2+u))+m32*(2*m44-2*(8*m_X2+2*s+2*t-3*u)*m42+2*m_X4+2*m_X2*(s+3*t-u)+u*(-2*s-2*t+u))+2*m2*m3*(2*m_X4-2*u*m_X2+u2+m42*(2*u-4*m_X2)))*m12+2*m4*(-2*m_X2*m44+u*m44+4*m_X2*s*m42+4*m_X2*t*m42-2*s*u*m42-2*t*u*m42-u3-2*m_X2*s2-2*m_X2*t2+2*m_X2*u2-4*m_X2*s*t-2*m_X4*u+s2*u+t2*u+2*s*t*u+2*m2*m3*(4*m_X4-2*u*m_X2+u2)+m32*(2*m_X4+(4*s+4*t-2*u)*m_X2+u*(-2*s-2*t+u)+m42*(2*u-4*m_X2))+m22*(2*m_X4+4*s*m_X2+4*t*m_X2-2*u*m_X2+u2-2*s*u-2*t*u+m42*(2*u-4*m_X2)+m32*(4*u-8*m_X2)))*m1-m48-2*m32*m46+2*m32*m42*m_X4-m42*u3-m44*s2+2*m_X4*s2-2*m42*m_X2*s2-m44*t2+2*m_X4*t2-2*m42*m_X2*t2+m44*u2+m32*m42*u2+2*m42*m_X2*u2+2*m46*s+2*m32*m44*s-2*m32*m_X4*s-2*m42*m_X4*s+2*m44*m_X2*s+6*m32*m42*m_X2*s+2*m46*t+2*m32*m44*t-2*m32*m_X4*t-2*m42*m_X4*t+2*m44*m_X2*t+2*m32*m42*m_X2*t-2*m44*s*t-4*m42*m_X2*s*t+m46*u+m32*m44*u-2*m44*m_X2*u-2*m32*m42*m_X2*u+m42*s2*u+m42*t2*u-2*m44*s*u-2*m32*m42*s*u-2*m44*t*u-2*m32*m42*t*u+2*m42*s*t*u+2*m2*m3*((2*m_X2-u)*m44+(2*m_X4-2*u*m_X2+u2)*m42-2*m_X4*u)+m22*(-2*m46+(2*s+2*t+u)*m44+(2*m_X4+2*(s+3*t-u)*m_X2+u*(-2*s-2*t+u))*m42-2*m_X4*(s+t)-4*m32*(m44+(2*m_X2-u)*m42-m_X4))) * u_propX

    tuX = (1/(m_X4))*2*(2*m18+(4*m22+4*m32+4*m42-5*s-3*t-3*u)*m16-2*(-2*m33-2*m4*m32+(-2*m42+s+t)*m3+m2*(2*m32+2*m42-4*m_X2-s)+m4*(-2*m42+s+u))*m15+((4*m32+4*m42-5*s-3*t-3*u)*m22+2*(-2*m33-2*m4*m32+(-2*m42+s+u)*m3+m4*(-2*m42+s+t))*m2+2*(m34+2*m4*m33+(2*m42-4*m_X2-3*s-t-u)*m32+m4*(2*m42-4*m_X2-s)*m3+m44+2*m_X2*s+2*(s+t)*(s+u)-m42*(4*m_X2+3*s+t+u)))*m14+(4*m35+4*m4*m34+(8*m42-4*(2*m_X2+2*s+t+u))*m33+4*m4*(2*m42-2*m_X2-2*s-t-u)*m32+(4*m44-4*(2*m_X2+2*s+t+u)*m42+3*s2+t2-u2+4*s*t+2*s*u+4*t*u+4*m_X2*(s-t+u))*m3+m22*(8*m33+8*m4*m32+8*m42*m3-4*(s+t)*m3+8*m43-4*m4*(s+u))+m4*(4*m44-4*(2*m_X2+2*s+t+u)*m42+3*s2-t2+u2+2*s*t+4*m_X2*(s+t-u)+4*s*u+4*t*u)-2*m2*(2*m34+(4*m42-4*m_X2-3*s-t-3*u)*m32+2*m44+(4*m_X2+s)*(s+t+u)-m42*(4*m_X2+3*s+3*t+u)))*m13+(4*m4*m35+(-8*m_X2-s+t+u)*m34+4*m4*(2*m42-2*s-t-u)*m33+((3*(-s+t+u)-32*m_X2)*m42+2*(4*m_X4+(7*s+5*(t+u))*m_X2+s2-t2-u2))*m32+m4*(4*m44-4*(2*s+t+u)*m42+8*m_X4+3*s2-t2-u2+8*m_X2*s+2*s*t+2*s*u+4*t*u)*m3+8*m42*m_X4-s3+t3+u3-8*m44*m_X2+2*m42*s2-4*m_X2*s2-2*m42*t2+s*t2-2*m42*u2+s*u2-t*u2-m44*s-8*m_X4*s+14*m42*m_X2*s+m44*t+10*m42*m_X2*t-s2*t-4*m_X2*s*t+m44*u+10*m42*m_X2*u-s2*u-t2*u-4*m_X2*s*u-2*s*t*u+m22*(8*m4*m33+(4*m42-24*m_X2-s+t+u)*m32+4*m4*(2*m42-4*m_X2-s)*m3+s2-t2-u2+12*m_X2*s+4*t*u+m42*(-24*m_X2-s+t+u))-2*m2*(2*m35+2*m4*m34+(4*m42-3*s-t-3*u)*m33+m4*(4*m42+8*m_X2-3*(s+t+u))*m32+(2*m44+(8*m_X2-3*(s+t+u))*m42-4*m_X4+s2+u2-2*m_X2*(s-t)+s*t+2*s*u)*m3+m4*(2*m44-(3*s+3*t+u)*m42-4*m_X4+s2+t2+2*s*t-2*m_X2*(s-u)+s*u)))*m12-(2*(2*m42+2*m_X2+s-t)*m35+2*m4*(2*m42+2*m_X2+s-t-u)*m34+(4*m44+2*(4*m_X2+s-3*(t+u))*m42-3*s2+3*t2+u2-2*s*u-8*m_X2*(s+u))*m33+m4*(4*m44+2*(4*m_X2+s-3*(t+u))*m42-8*m_X4-4*m_X2*(3*s+t)+3*(-s2+t2+u2))*m32+(2*(2*m_X2+s-t-u)*m44-(8*m_X4+4*(3*s+u)*m_X2+3*(s2-t2-u2))*m42+s3-t3-u3-s*t2-s*u2+t*u2+8*m_X4*t+s2*t+s2*u+t2*u+2*s*t*u+4*m_X2*(s2+u*s-t2+u2-t*u))*m3+m4*(2*(2*m_X2+s-u)*m44+(-8*(s+t)*m_X2-3*s2+t2+3*u2-2*s*t)*m42+s3-t3-u3-s*t2-s*u2+t*u2+s2*t+8*m_X4*u+s2*u+t2*u+2*s*t*u+4*m_X2*(s2+t*s+t2-u2-t*u))+m22*(2*(4*m_X2+s-t+u)*m33+2*m4*(12*m_X2+s-t-u)*m32+(-8*m_X4+(4*t-8*s)*m_X2-s2+t2+u2+2*m42*(12*m_X2+s-t-u)-4*t*u)*m3+m4*(-8*m_X4+(4*u-8*s)*m_X2-s2+t2+u2+2*m42*(4*m_X2+s+t-u)-4*t*u))+2*m2*(4*m4*m35+(2*m42+2*m_X2-u)*m34+4*m4*(2*m42-s-t-u)*m33+(2*m44+(8*m_X2-s-3*(t+u))*m42-4*m_X4+u*(s+u)+2*m_X2*(-s+t+u))*m32+m4*(4*m44-4*(s+t+u)*m42-16*m_X4+s2+t2+u2+2*s*t+2*s*u+4*m_X2*(t+u))*m3+m44*(2*m_X2-t)+2*m_X2*(2*s*m_X2-2*t*u-s*(t+u))+m42*(-4*m_X4+2*(-s+t+u)*m_X2+t*(s+t))))*m1-4*m33*m45-2*m34*m44-4*m35*m43-m3*m4*s3+m3*m4*t3+m3*m4*u3+4*m3*m45*m_X2+4*m35*m4*m_X2+8*m_X4*s2+3*m3*m43*s2+m32*m42*s2-2*m32*m_X2*s2-2*m42*m_X2*s2+3*m33*m4*s2-m3*m43*t2-m32*m42*t2+2*m32*m_X2*t2-2*m42*m_X2*t2-3*m33*m4*t2+m3*m4*s*t2-3*m3*m43*u2-m32*m42*u2-2*m32*m_X2*u2+2*m42*m_X2*u2-m33*m4*u2+m3*m4*s*u2-m3*m4*t*u2-2*m3*m45*s-m32*m44*s-8*m32*m_X4*s-8*m42*m_X4*s-8*m3*m4*m_X4*s-2*m33*m43*s-m34*m42*s+2*m34*m_X2*s+2*m44*m_X2*s-4*m3*m43*m_X2*s+12*m32*m42*m_X2*s-4*m33*m4*m_X2*s-2*m35*m4*s+m32*m44*t+6*m33*m43*t+m34*m42*t-2*m34*m_X2*t+2*m44*m_X2*t-4*m3*m43*m_X2*t-4*m33*m4*m_X2*t-m3*m4*s2*t+2*m35*m4*t+2*m3*m43*s*t-4*m42*m_X2*s*t+4*m3*m4*m_X2*s*t+2*m3*m45*u+m32*m44*u+6*m33*m43*u+m34*m42*u+2*m34*m_X2*u-2*m44*m_X2*u-4*m3*m43*m_X2*u-4*m33*m4*m_X2*u-m3*m4*s2*u-m3*m4*t2*u-4*m32*m_X2*s*u+4*m3*m4*m_X2*s*u+2*m33*m4*s*u+8*m3*m4*m_X2*t*u-2*m3*m4*s*t*u+m22*(-8*s*m_X4+2*m42*(4*m_X2+s+3*t-u)*m_X2+2*m33*m4*(4*m_X2-s+t-u)+m3*m4*(8*m_X4-8*(t+u)*m_X2+s2-t2-u2+4*t*u+2*m42*(4*m_X2-s-t+u))+m32*((-16*m_X2-s+t+u)*m42+2*m_X2*(4*m_X2+s-t+3*u)))-2*m2*(2*(m42-m_X2)*m35+m4*(2*m42-2*m_X2-u)*m34+(2*m44-(4*m_X2+s+3*t+u)*m42+2*m_X2*(s+t+u))*m33+m4*(2*m44-(4*m_X2+s+t+3*u)*m42-4*m_X4+u*(s+u)+2*m_X2*(t+2*u))*m32+(-((2*m_X2+t)*m44)+(-4*m_X4+2*(2*t+u)*m_X2+t*(s+t))*m42+2*m_X2*(2*u*m_X2-s*t-t*u))*m3+2*m4*m_X2*(-m44+(s+t+u)*m42+2*m_X2*t-(s+t)*u))) * t_propX*u_propX*((t-m_X2)*(u-m_X2)+m_Gamma_X2)

    thuX = (1/(m_X4))*8*m_d2*(2*(m2-m4)*m15+(-4*m42-2*m3*m4+2*m2*m3+s+t-u)*m14-2*(2*m4*m22+(-2*m32+s+t+u)*m2+2*m4*(m32+m42-s-t))*m13-(4*m44-4*m_X2*m42-6*s*m42-6*t*m42-2*u*m42+2*m3*(2*m_X2-s-t)*m4+s2+t2-u2+4*m_X2*t+2*s*t+m32*(8*m42-s-3*t+u)+m22*(4*m32+4*m4*m3+8*m42-4*m_X2-3*s-t+u)+2*m2*(m4*(u-4*m_X2)+m3*(2*m42-2*m_X2+u)))*m12-2*((4*m4*m32+(u-4*m_X2)*m3+m4*(2*m42-2*m_X2-2*s-2*t+u))*m22+(m44+(2*m_X2-s-t)*m42+2*m3*(u-4*m_X2)*m4-2*m_X2*s-t*u+m32*(2*m42+2*m_X2+u))*m2-m3*(m42-t)*(4*m_X2-u)+m32*m4*(2*m42-2*m_X2-2*s-2*t+u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+2*s*t+2*m_X2*u))*m1+2*m3*m45+4*m32*m42*m_X2-m42*s2-m42*t2+4*m_X2*t2+m42*u2+m44*s-2*m3*m43*s+3*m32*m42*s+4*m3*m4*m_X2*s+m44*t-2*m3*m43*t+m32*m42*t-4*m32*m_X2*t-4*m42*m_X2*t-2*m42*s*t-m44*u-2*m3*m43*u-m32*m42*u+2*m3*m4*t*u+2*m2*(m4*(4*m_X2-u)*m32+(m42+2*m_X2)*(m42-u)*m3+m4*t*(u-4*m_X2))+m22*(4*m3*m43+(s+3*t-u)*m42-2*m3*(2*m_X2+u)*m4-4*m32*(m42-m_X2)-4*m_X2*t)) * u_propX*(-t_proph)*((u-m_X2)*(t-m_h2)+m_Gamma_X*m_Gamma_h)

    uhtX = (1/(m_X4))*8*m_d2*(2*(m2-m3)*m15+(-4*m32-2*m4*m3+2*m2*m4+s-t+u)*m14-2*(2*m3*m22+(-2*m42+s+t+u)*m2+2*m3*(m32+m42-s-u))*m13-(4*m34+(8*m42-2*(2*m_X2+3*s+t+3*u))*m32+2*m4*(2*m_X2-s-u)*m3+s2-t2+u2-m42*s+m42*t+2*m2*(2*m4*m32+(t-4*m_X2)*m3+m4*(t-2*m_X2))+m22*(8*m32+4*m4*m3+4*m42-4*m_X2-3*s+t-u)-3*m42*u+4*m_X2*u+2*s*u)*m12-2*(m35+2*(m42-s-u)*m33+m4*(t-4*m_X2)*m32+((-2*m_X2-2*s+t-2*u)*m42+s2-t2+u2+2*m_X2*t+2*s*u)*m3+m22*(2*m33+(4*m42-2*m_X2-2*s+t-2*u)*m3+m4*(t-4*m_X2))+m4*(4*m_X2-t)*u+m2*(m34+(2*m42+2*m_X2-s-u)*m32+2*m4*(t-4*m_X2)*m3-2*m_X2*s+m42*(2*m_X2+t)-t*u))*m1+4*m32*m42*m_X2-m32*s2+m32*t2-m32*u2+4*m_X2*u2+2*m35*m4+m34*s+3*m32*m42*s+4*m3*m4*m_X2*s-2*m33*m4*s-m34*t-m32*m42*t-2*m33*m4*t+2*m2*(m4*m34+m4*(2*m_X2-t)*m32+(4*m_X2-t)*(m42-u)*m3-2*m4*m_X2*t)+m34*u+m32*m42*u-4*m32*m_X2*u-4*m42*m_X2*u-2*m33*m4*u-2*m32*s*u+2*m3*m4*t*u+m22*(4*m4*m33+(-4*m42+s-t+3*u)*m32-2*m4*(2*m_X2+t)*m3+4*m_X2*(m42-u))) * t_propX*(-u_proph)*((t-m_X2)*(u-m_h2)+m_Gamma_X*m_Gamma_h)

    X_squared = ssX + ttX + uuX + stX + suX + tuX 
    h_squared = ssh + tth + uuh + sth + suh + tuh
    Xh_interference = shtX + shuX + thsX + thuX + uhsX + uhtX

    return vert*(X_squared + h_squared + Xh_interference)

@nb.jit(nopython=True, cache=True)
def M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False):
    """
    Anton: 
    12 --> 34, 1,2,3,4 = a, d
    sub = True: |D_off-shell|^2 is used -- on-shell contribution is subtracted
    sub = False: |D_BW|^2 is used
    """
    m12 = m1*m1
    m13 = m1*m12
    m14 = m12*m12
    m15 = m12*m13
    m16 = m12*m14
    # m17 = m13*m14
    m18 = m14*m14

    m22 = m2*m2
    m23 = m2*m22
    m24 = m22*m22
    m25 = m22*m23
    m26 = m22*m24
    # m27 = m23*m24
    m28 = m24*m24

    m32 = m3*m3
    m33 = m3*m32
    m34 = m32*m32
    m35 = m32*m33
    # m36 = m32*m34
    # m37 = m33*m34
    # m38 = m34*m34

    m42 = m4*m4
    m43 = m4*m42
    m44 = m42*m42
    m45 = m42*m43
    # m46 = m42*m44
    # m47 = m43*m44
    # m48 = m44*m44

    m_X4 = m_X2*m_X2
    u = m12 + m22 + m32 + m42 - s - t

    s2 = s*s
    s3 = s*s2
    t2 = t*t
    t3 = t*t2
    u2 = u*u
    u3 = u*u2

    """
    Anton: 
    https://arxiv.org/pdf/2309.16615
    Subtract on-shell contribution from Breit-Wigner propagator (RIS-subtraction) to 
    avoid double counting decay processes
    Goal: D_BW --> D_off-shell, |D_BW|^2 --> |D_off-shell|^2
    D_BW(s) = 1 / (s - m^2 + imG) = (s - m^2)/((s-m^2)^2 + (mG)^2) - imG/((s-m^2)^2 + (mG)^2)
    |D_BW(s)|^2 = 1 / ((s-m^2)^2 + (mG)^2) := s_prop
    D_BW(s) = (s-m^2)*s_prop - imG*s_prop
    The real part of D_BW is defined as the off-shell propagator 
    D_off-shell(s) := Re(D_BW(s)) = (s-m^2)*s_prop -- used for interference terms st, su
    D_off-shell(t) := Re(D_BW(t)) = (t-m^2)*t_prop -- used for interference term st
    D_off-shell(u) := Re(D_BW(u)) = (u-m^2)*u_prop -- used for interference term su
    Need another expression for squared off-shell propagator:
    |D_off-shell(s)|^2 := ((s - m^2)^2 - (mG)^2) / ((s-m^2)^2 + (mG)^2)^2 = ((s - m^2)^2 - (mG)^2)*s_prop*s_prop
    -- used in ss
    Also recall: |M|^2 contains only 2*Re(D(s)* x D(t)) etc. for cross-terms, so only the real part.
    Re(D(s)* x D(t)) = s_prop*t_prop*(s - m_X2)*(t - m_X2) + (0 if sub else m_GammaX2)
    |D(s)|^2 = s_prop*(s_prop*(s - m_X2)^2 - m_Gamma_X2 if sub else 1.) 
    EDIT: Numba does not like one-line if-else tests (like written above for propagator). Split 
    test into if-else blocks.
    """
    # Anton: Squared BW-propagators, |D_BW|^2
    s_prop = 1. / ((s - m_X2)*(s - m_X2) + m_Gamma_X2)
    t_prop = 1. / ((t - m_X2)*(t - m_X2) + m_Gamma_X2)
    u_prop = 1. / ((u - m_X2)*(u - m_X2) + m_Gamma_X2)
    
    # Anton: For s-channel processes, need to take care of real intermediate state subtraction (RIS)
    if sub: 
        ss = 1/m_X4*4*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+(4*m_X2*m2-2*m2*s)*m15+(2*m24+(2*m32+2*m42+3*s-2*t-2*u)*m22-4*m32*m42+s2-t2-u2+m32*s+m42*s+2*m3*m4*s+2*m32*t+2*m42*t-2*s*t-2*m_X2*(4*m22+2*m3*m4+s-t-u)+2*m32*u+2*m42*u-2*s*u-2*t*u)*m14+4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+(2*(2*m22+m32+m42-2*m3*m4-t-u)*m_X4-2*(4*m24+(8*m32-4*m4*m3+8*m42-2*(s+3*(t+u)))*m22-s2+t2+u2+m42*s-2*m3*m4*s-m42*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m_X2+m24*(2*m32+2*m42+3*s-2*t-2*u)+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u)))+2*m22*((4*m42+3*s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2-2*s*t-2*s*u+2*t*u+m42*(3*s-2*(t+u))))*m12-2*m2*(2*(m32-4*m4*m3+m42-s)*m_X4-2*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u)))*m_X2+s*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32+2*m4*s*m3+t2+u2-m42*(t+u))+m24*(-2*(2*m3*m4+s-t-u)*m_X2+s2-t2-u2+m42*s+2*m3*m4*s+2*m42*t-2*s*t+2*m42*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))+m22*(2*(m32-2*m4*m3+m42-t-u)*m_X4-2*((4*m42+s-t-3*u)*m32-2*m4*s*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m_X2+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u))))) * s_prop*s_prop*((s-m_X2)*(s-m_X2) - m_Gamma_X2)            # Anton: Square off-shell propagator 

        st = 1/m_X4*2*(2*m18+(4*m22+4*m32+4*m42-3*s-3*t-5*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m32+s+u)*m2-2*m33-2*m32*m4+m4*(4*m_X2+u)+m3*(t+u))*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+t+3*u))*m22+(4*m33+4*m4*m32-2*u*m3-2*m4*(t+u))*m2+2*m34+4*m32*m42+4*u2+4*m33*m4-2*m32*s-3*m42*s-2*m3*m4*s-2*m32*t-3*m42*t+4*s*t-4*m_X2*(2*m22+2*m3*m2+2*m32-u)-6*m32*u-5*m42*u-2*m3*m4*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+t+2*u)*m23+(-8*m33-8*m4*m32+4*(-2*m42+s+t+2*u)*m3+2*m4*(s+3*(t+u)))*m22+(-4*m34+4*(-2*m42+s+t+2*u)*m32-s2+t2-3*u2-4*s*t-4*s*u-2*t*u+4*m42*(s+u))*m2-4*m35-8*m33*m42+m3*s2-m3*t2-3*m3*u2-2*m4*u2-4*m34*m4+4*m33*s+6*m32*m4*s+4*m33*t+4*m3*m42*t+2*m32*m4*t-4*m3*s*t+8*m33*u+4*m3*m42*u+6*m32*m4*u-2*m3*s*u-2*m4*s*u-4*m3*t*u-2*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m32+s-t-u)*m2+2*m33+2*m32*m4-m3*(s-t+u)-2*m4*(s+t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s+t-u)*m24+(8*m33+8*m4*m32+(8*m42-4*(s+t+2*u))*m3-2*m4*(s+3*(t+u)))*m23+(8*m4*m33+(4*m42+3*(s+t-u))*m32-6*m4*(s+t+u)*m3+m42*(s+t-u)-2*(s2+t2-u2))*m22+(4*m35+4*m4*m34+(8*m42-4*(s+t+2*u))*m33-6*m4*(s+t+u)*m32-(4*u*m42+s2+t2-3*u2-2*t*u-2*s*(2*t+u))*m3+2*m4*(t2+2*u*t+u*(s+u)))*m2+s3+t3-u3-2*m32*s2-m42*s2+2*m3*m4*s2-2*m32*t2-m42*t2-s*t2+2*m32*u2+m42*u2+2*m3*m4*u2-s*u2-t*u2+4*m35*m4+m34*s+m32*m42*s-6*m33*m4*s+m34*t+m32*m42*t-s2*t-2*m33*m4*t+4*m42*s*t+8*m_X4*(m22+(m3-m4)*m2+m32-m3*m4-u)-m34*u-m32*m42*u+s2*u+t2*u-6*m33*m4*u+4*m3*m4*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(16*m32-8*m4*m3+12*m42-5*s-5*t-7*u)*m22-2*(4*m4*m32-4*m42*m3+2*u*m3+m4*s-m4*u)*m2+4*m34+m32*(12*m42-5*s-5*t-7*u)+2*m3*m4*(u-t)+2*u*(-3*m42+s+t+u)))*m12+(2*(2*m32-4*m4*m3-s+u)*m25+2*(2*m33-2*m4*m32-(s+t-u)*m3+m4*t)*m24+(4*m34-16*m4*m33+(-6*s-6*t+2*u)*m32+8*m4*(s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m23+(4*m35-4*m4*m34+(-6*s-6*t+2*u)*m33+2*m4*(3*s+3*t+u)*m32-2*m42*(s+t-u)*m3+3*(s2+t2-u2)*m3-2*m4*t*(t+u))*m22+(-8*m4*m35-2*(s+t-u)*m34+8*m4*(s+t+u)*m33+(3*(s2+t2-u2)-2*m42*(s+t-u))*m32-2*m4*(s2+2*u*s+(t+u)**2)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))*m2-8*m_X4*((m3-m4)*m22+(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m3*(-2*(t-u)*m34+2*m4*s*m33+(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u)*m32-2*m4*s*(s+u)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))+4*m_X2*(m25+(m3-m4)*m24+2*(m32+m42-t-u)*m23+(2*m33-4*m4*m32+(6*m42-s-3*u)*m3-m4*(s+t-u))*m22+(m34+(6*m42-t-3*u)*m32-2*m4*(s+t)*m3-s2+t2+u2-s*t+m42*(s-2*u)+t*u)*m2+m35-m34*m4+2*m33*(m42-s-u)-m32*m4*(s+t-u)+m3*((t-2*u)*m42+s2-t2+u2+s*(u-t))+m4*(t*u+s*(2*t+u))))*m1-8*m_X4*((-m42+m3*m4+u)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2-m3*m4*s+(m42-u)*u+m32*(u-m42))-m2*m3*(2*(2*m32-2*m4*m3-s+u)*m24+(2*m33-4*m4*m32-(s+t-u)*m3+2*m4*t)*m23+(4*m34-4*m4*m33+(-6*s-6*t+2*u)*m32+2*m4*(3*s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m22-(4*m4*m34+(s+t-u)*m33-2*m4*(s+3*t+u)*m32+((s+t-u)*m42-s2-t2+u2)*m3+2*m4*t*(t+u))*m2-s3-t3+u3+m42*s2+m42*t2+s*t2-m42*u2+s*u2+t*u2+2*m33*m4*s+s2*t-4*m42*s*t-2*m34*(t-u)-s2*u-t2*u+2*s*t*u-2*m3*m4*s*(s+u)+m32*(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u))+2*m_X2*(2*(m3-m4)*m25+(-2*m3*m4-s+t+u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23-(4*m4*m33+(8*m42-6*u)*m32-2*m4*(s+2*t)*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m22+2*(m35-m4*m34+(2*m42-s-t-u)*m33+m4*(2*s+t)*m32+(-2*(s+t)*m42+2*s*t+s*u+t*u)*m3-m4*s*(t+u))*m2+m3*(-2*m4*m34+(s-t+u)*m33+2*m4*(s+t+u)*m32+((3*s-t+u)*m42-s2+t2-u2-2*s*u)*m3-2*m4*t*(s+u)))) * s_prop*t_prop*(s-m_X2)*(t-m_X2)           # Anton: Interference off-shell propagator 

        su = -1/m_X4*2*(2*m18+(4*m22+4*m32+4*m42-3*s-5*t-3*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m42+s+t)*m2-2*m43-2*m3*m42+4*m_X2*m3+m3*t+m4*t+m4*u)*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+3*t+u))*m22+(4*m43+4*m3*m42-2*t*m4-2*m3*(t+u))*m2+2*m44+4*m3*m43+4*m32*m42+4*t2-3*m32*s-2*m42*s-2*m3*m4*s-4*m_X2*(2*m22+2*m4*m2+2*m42-t)-5*m32*t-6*m42*t-2*m3*m4*t+4*s*t-3*m32*u-2*m42*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+2*t+u)*m23+(-8*m4*m32+2*(-4*m42+s+3*(t+u))*m3+4*m4*(-2*m42+s+2*t+u))*m22+(-4*m44+4*(s+2*t+u)*m42-s2-3*t2+u2-4*s*t+4*m32*(-2*m42+s+t)-4*s*u-2*t*u)*m2-4*m45-4*m3*m44-8*m32*m43+m4*s2-2*m3*t2-3*m4*t2-m4*u2+4*m43*s+6*m3*m42*s+8*m43*t+6*m3*m42*t+4*m32*m4*t-2*m3*s*t-2*m4*s*t+4*m43*u+2*m3*m42*u+4*m32*m4*u-4*m4*s*u-2*m3*t*u-4*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m42+s-t-u)*m2+2*m3*(m42-s-t-u)+m4*(2*m42-s-t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s-t+u)*m24+(8*m4*m32+8*m42*m3-2*(s+3*(t+u))*m3+4*m4*(2*m42-s-2*t-u))*m23+(8*m3*m43+3*(s-t+u)*m42-6*m3*(s+t+u)*m4+m32*(4*m42+s-t+u)-2*(s2-t2+u2))*m22+((8*m43-4*m4*t)*m32+2*(2*m44-3*(s+t+u)*m42+(t+u)**2+s*t)*m3+m4*(4*m44-4*(s+2*t+u)*m42-s2+3*t2-u2+2*t*u+2*s*(t+2*u)))*m2+4*m3*m45+s3-t3+u3-m32*s2-2*m42*s2+2*m3*m4*s2+m32*t2+2*m42*t2+2*m3*m4*t2-s*t2-m32*u2-2*m42*u2-s*u2+t*u2+m44*s-6*m3*m43*s+m32*m42*s+8*m_X4*(m22+(m4-m3)*m2+m42-m3*m4-t)-m44*t-6*m3*m43*t-m32*m42*t+s2*t+4*m3*m4*s*t+m44*u-2*m3*m43*u+m32*m42*u-s2*u-t2*u+4*m32*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(12*m32-8*m4*m3+16*m42-5*s-7*t-5*u)*m22+2*(4*m4*m32+(-4*m42-s+t)*m3-2*m4*t)*m2+4*m44+2*t2-5*m42*s+6*m32*(2*m42-t)-7*m42*t+2*s*t+2*m3*m4*(t-u)-5*m42*u+2*t*u))*m12+(-2*(-2*m42+4*m3*m4+s-t)*m25-2*(m3*(2*m42-u)+m4*(-2*m42+s-t+u))*m24+(4*m44+(-6*s+2*t-6*u)*m42+8*m3*(-2*m42+s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m23+(4*m45+(-6*s+2*t-6*u)*m43-2*m32*(s-t+u)*m4+3*(s2-t2+u2)*m4-2*m3*(2*m44-(3*s+t+3*u)*m42+u*(t+u)))*m22+(-2*(s-t+u)*m44+3*(s2-t2+u2)*m42-2*m3*(4*m44-4*(s+t+u)*m42+s2+(t+u)**2+2*s*t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(-2*(s-t+u)*m42+s2-t2+u2-4*s*u))*m2+8*m_X4*((m3-m4)*m22-(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m4*(2*(t-u)*m44+(s2-2*t*s-3*t2+3*u2)*m42+2*m3*s*(m42-s-t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u))+4*m_X2*(m25+(m4-m3)*m24+2*(m32+m42-t-u)*m23+(6*m4*m32-(4*m42+s-t+u)*m3+m4*(2*m42-s-3*t))*m22+(m44-(3*t+u)*m42-2*m3*(s+u)*m4-s2+t2+u2+m32*(6*m42+s-2*t)-s*u+t*u)*m2+m32*m4*(2*m42-2*t+u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+s*(t-u))+m3*(-m44-(s-t+u)*m42+t*u+s*(t+2*u))))*m1+8*m_X4*((m32-m4*m3-t)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2+m3*m4*s+m32*(m42-t)+t*(t-m42))-2*m_X2*(2*(m3-m4)*m25+(2*m3*m4+s-t-u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23+(4*m3*m43-6*t*m42-2*m3*(s+2*u)*m4-s2+t2+u2+m32*(8*m42+s-t-3*u)+2*t*u)*m22-2*(2*m4*(m42-s-u)*m32-(m44-(2*s+u)*m42+s*(t+u))*m3+m4*(m44-(s+t+u)*m42+s*t+2*s*u+t*u))*m2+m4*(m4*(-3*s-t+u)*m32+2*(m42-s-t)*(m42-u)*m3-m4*(m42-s-t-u)*(s+t-u)))-m2*m4*(-2*(-2*m42+2*m3*m4+s-t)*m24+(m4*(2*m42-s+t-u)+m3*(2*u-4*m42))*m23+(4*m44+(-6*s+2*t-6*u)*m42+2*m3*(-2*m42+3*s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m22-(m4*(s-t+u)*m32+2*(2*m44-(s+t+3*u)*m42+u*(t+u))*m3+m4*((s-t+u)*m42-s2+t2-u2))*m2-s3+t3-u3+m42*s2-3*m42*t2+s*t2+3*m42*u2+s*u2-t*u2+2*m3*m4*s*(m42-s-t)+2*m44*t-s2*t-2*m42*s*t-2*m44*u+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u))) * s_prop*u_prop*(s-m_X2)*(u-m_X2)

    else: 
        ss = 1/m_X4*4*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+(4*m_X2*m2-2*m2*s)*m15+(2*m24+(2*m32+2*m42+3*s-2*t-2*u)*m22-4*m32*m42+s2-t2-u2+m32*s+m42*s+2*m3*m4*s+2*m32*t+2*m42*t-2*s*t-2*m_X2*(4*m22+2*m3*m4+s-t-u)+2*m32*u+2*m42*u-2*s*u-2*t*u)*m14+4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+(2*(2*m22+m32+m42-2*m3*m4-t-u)*m_X4-2*(4*m24+(8*m32-4*m4*m3+8*m42-2*(s+3*(t+u)))*m22-s2+t2+u2+m42*s-2*m3*m4*s-m42*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m_X2+m24*(2*m32+2*m42+3*s-2*t-2*u)+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u)))+2*m22*((4*m42+3*s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2-2*s*t-2*s*u+2*t*u+m42*(3*s-2*(t+u))))*m12-2*m2*(2*(m32-4*m4*m3+m42-s)*m_X4-2*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u)))*m_X2+s*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32+2*m4*s*m3+t2+u2-m42*(t+u))+m24*(-2*(2*m3*m4+s-t-u)*m_X2+s2-t2-u2+m42*s+2*m3*m4*s+2*m42*t-2*s*t+2*m42*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))+m22*(2*(m32-2*m4*m3+m42-t-u)*m_X4-2*((4*m42+s-t-3*u)*m32-2*m4*s*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m_X2+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u))))) * s_prop

        st = 1/m_X4*2*(2*m18+(4*m22+4*m32+4*m42-3*s-3*t-5*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m32+s+u)*m2-2*m33-2*m32*m4+m4*(4*m_X2+u)+m3*(t+u))*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+t+3*u))*m22+(4*m33+4*m4*m32-2*u*m3-2*m4*(t+u))*m2+2*m34+4*m32*m42+4*u2+4*m33*m4-2*m32*s-3*m42*s-2*m3*m4*s-2*m32*t-3*m42*t+4*s*t-4*m_X2*(2*m22+2*m3*m2+2*m32-u)-6*m32*u-5*m42*u-2*m3*m4*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+t+2*u)*m23+(-8*m33-8*m4*m32+4*(-2*m42+s+t+2*u)*m3+2*m4*(s+3*(t+u)))*m22+(-4*m34+4*(-2*m42+s+t+2*u)*m32-s2+t2-3*u2-4*s*t-4*s*u-2*t*u+4*m42*(s+u))*m2-4*m35-8*m33*m42+m3*s2-m3*t2-3*m3*u2-2*m4*u2-4*m34*m4+4*m33*s+6*m32*m4*s+4*m33*t+4*m3*m42*t+2*m32*m4*t-4*m3*s*t+8*m33*u+4*m3*m42*u+6*m32*m4*u-2*m3*s*u-2*m4*s*u-4*m3*t*u-2*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m32+s-t-u)*m2+2*m33+2*m32*m4-m3*(s-t+u)-2*m4*(s+t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s+t-u)*m24+(8*m33+8*m4*m32+(8*m42-4*(s+t+2*u))*m3-2*m4*(s+3*(t+u)))*m23+(8*m4*m33+(4*m42+3*(s+t-u))*m32-6*m4*(s+t+u)*m3+m42*(s+t-u)-2*(s2+t2-u2))*m22+(4*m35+4*m4*m34+(8*m42-4*(s+t+2*u))*m33-6*m4*(s+t+u)*m32-(4*u*m42+s2+t2-3*u2-2*t*u-2*s*(2*t+u))*m3+2*m4*(t2+2*u*t+u*(s+u)))*m2+s3+t3-u3-2*m32*s2-m42*s2+2*m3*m4*s2-2*m32*t2-m42*t2-s*t2+2*m32*u2+m42*u2+2*m3*m4*u2-s*u2-t*u2+4*m35*m4+m34*s+m32*m42*s-6*m33*m4*s+m34*t+m32*m42*t-s2*t-2*m33*m4*t+4*m42*s*t+8*m_X4*(m22+(m3-m4)*m2+m32-m3*m4-u)-m34*u-m32*m42*u+s2*u+t2*u-6*m33*m4*u+4*m3*m4*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(16*m32-8*m4*m3+12*m42-5*s-5*t-7*u)*m22-2*(4*m4*m32-4*m42*m3+2*u*m3+m4*s-m4*u)*m2+4*m34+m32*(12*m42-5*s-5*t-7*u)+2*m3*m4*(u-t)+2*u*(-3*m42+s+t+u)))*m12+(2*(2*m32-4*m4*m3-s+u)*m25+2*(2*m33-2*m4*m32-(s+t-u)*m3+m4*t)*m24+(4*m34-16*m4*m33+(-6*s-6*t+2*u)*m32+8*m4*(s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m23+(4*m35-4*m4*m34+(-6*s-6*t+2*u)*m33+2*m4*(3*s+3*t+u)*m32-2*m42*(s+t-u)*m3+3*(s2+t2-u2)*m3-2*m4*t*(t+u))*m22+(-8*m4*m35-2*(s+t-u)*m34+8*m4*(s+t+u)*m33+(3*(s2+t2-u2)-2*m42*(s+t-u))*m32-2*m4*(s2+2*u*s+(t+u)**2)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))*m2-8*m_X4*((m3-m4)*m22+(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m3*(-2*(t-u)*m34+2*m4*s*m33+(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u)*m32-2*m4*s*(s+u)*m3-s3-t3+u3+s*t2+s*u2+t*u2+s2*t-s2*u-t2*u+2*s*t*u+m42*(s2-4*t*s+t2-u2))+4*m_X2*(m25+(m3-m4)*m24+2*(m32+m42-t-u)*m23+(2*m33-4*m4*m32+(6*m42-s-3*u)*m3-m4*(s+t-u))*m22+(m34+(6*m42-t-3*u)*m32-2*m4*(s+t)*m3-s2+t2+u2-s*t+m42*(s-2*u)+t*u)*m2+m35-m34*m4+2*m33*(m42-s-u)-m32*m4*(s+t-u)+m3*((t-2*u)*m42+s2-t2+u2+s*(u-t))+m4*(t*u+s*(2*t+u))))*m1-8*m_X4*((-m42+m3*m4+u)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2-m3*m4*s+(m42-u)*u+m32*(u-m42))-m2*m3*(2*(2*m32-2*m4*m3-s+u)*m24+(2*m33-4*m4*m32-(s+t-u)*m3+2*m4*t)*m23+(4*m34-4*m4*m33+(-6*s-6*t+2*u)*m32+2*m4*(3*s+t+u)*m3+3*s2+t2-3*u2-2*t*u+2*m42*(-s+t+u))*m22-(4*m4*m34+(s+t-u)*m33-2*m4*(s+3*t+u)*m32+((s+t-u)*m42-s2-t2+u2)*m3+2*m4*t*(t+u))*m2-s3-t3+u3+m42*s2+m42*t2+s*t2-m42*u2+s*u2+t*u2+2*m33*m4*s+s2*t-4*m42*s*t-2*m34*(t-u)-s2*u-t2*u+2*s*t*u-2*m3*m4*s*(s+u)+m32*(2*(s-t+u)*m42+s2+3*t2-3*u2-2*s*u))+2*m_X2*(2*(m3-m4)*m25+(-2*m3*m4-s+t+u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23-(4*m4*m33+(8*m42-6*u)*m32-2*m4*(s+2*t)*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m22+2*(m35-m4*m34+(2*m42-s-t-u)*m33+m4*(2*s+t)*m32+(-2*(s+t)*m42+2*s*t+s*u+t*u)*m3-m4*s*(t+u))*m2+m3*(-2*m4*m34+(s-t+u)*m33+2*m4*(s+t+u)*m32+((3*s-t+u)*m42-s2+t2-u2-2*s*u)*m3-2*m4*t*(s+u)))) * s_prop*t_prop*((s-m_X2)*(t-m_X2) + m_Gamma_X2)

        su = -1/m_X4*2*(2*m18+(4*m22+4*m32+4*m42-3*s-5*t-3*u)*m16+2*(-2*m23-2*(m3+m4)*m22+(-2*m42+s+t)*m2-2*m43-2*m3*m42+4*m_X2*m3+m3*t+m4*t+m4*u)*m15+(2*m24+4*(m3+m4)*m23+(4*m32+4*m4*m3+4*m42-2*(s+3*t+u))*m22+(4*m43+4*m3*m42-2*t*m4-2*m3*(t+u))*m2+2*m44+4*m3*m43+4*m32*m42+4*t2-3*m32*s-2*m42*s-2*m3*m4*s-4*m_X2*(2*m22+2*m4*m2+2*m42-t)-5*m32*t-6*m42*t-2*m3*m4*t+4*s*t-3*m32*u-2*m42*u+4*s*u+4*t*u)*m14+(-4*m25-4*(m3+m4)*m24+4*(-2*m32-2*m42+s+2*t+u)*m23+(-8*m4*m32+2*(-4*m42+s+3*(t+u))*m3+4*m4*(-2*m42+s+2*t+u))*m22+(-4*m44+4*(s+2*t+u)*m42-s2-3*t2+u2-4*s*t+4*m32*(-2*m42+s+t)-4*s*u-2*t*u)*m2-4*m45-4*m3*m44-8*m32*m43+m4*s2-2*m3*t2-3*m4*t2-m4*u2+4*m43*s+6*m3*m42*s+8*m43*t+6*m3*m42*t+4*m32*m4*t-2*m3*s*t-2*m4*s*t+4*m43*u+2*m3*m42*u+4*m32*m4*u-4*m4*s*u-2*m3*t*u-4*m4*t*u+4*m_X2*(2*m23+2*(m3+m4)*m22+(2*m42+s-t-u)*m2+2*m3*(m42-s-t-u)+m4*(2*m42-s-t+u)))*m13+(4*(m3+m4)*m25+(4*m3*m4+s-t+u)*m24+(8*m4*m32+8*m42*m3-2*(s+3*(t+u))*m3+4*m4*(2*m42-s-2*t-u))*m23+(8*m3*m43+3*(s-t+u)*m42-6*m3*(s+t+u)*m4+m32*(4*m42+s-t+u)-2*(s2-t2+u2))*m22+((8*m43-4*m4*t)*m32+2*(2*m44-3*(s+t+u)*m42+(t+u)**2+s*t)*m3+m4*(4*m44-4*(s+2*t+u)*m42-s2+3*t2-u2+2*t*u+2*s*(t+2*u)))*m2+4*m3*m45+s3-t3+u3-m32*s2-2*m42*s2+2*m3*m4*s2+m32*t2+2*m42*t2+2*m3*m4*t2-s*t2-m32*u2-2*m42*u2-s*u2+t*u2+m44*s-6*m3*m43*s+m32*m42*s+8*m_X4*(m22+(m4-m3)*m2+m42-m3*m4-t)-m44*t-6*m3*m43*t-m32*m42*t+s2*t+4*m3*m4*s*t+m44*u-2*m3*m43*u+m32*m42*u-s2*u-t2*u+4*m32*s*u+2*m3*m4*t*u-2*s*t*u-2*m_X2*(4*m24+(12*m32-8*m4*m3+16*m42-5*s-7*t-5*u)*m22+2*(4*m4*m32+(-4*m42-s+t)*m3-2*m4*t)*m2+4*m44+2*t2-5*m42*s+6*m32*(2*m42-t)-7*m42*t+2*s*t+2*m3*m4*(t-u)-5*m42*u+2*t*u))*m12+(-2*(-2*m42+4*m3*m4+s-t)*m25-2*(m3*(2*m42-u)+m4*(-2*m42+s-t+u))*m24+(4*m44+(-6*s+2*t-6*u)*m42+8*m3*(-2*m42+s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m23+(4*m45+(-6*s+2*t-6*u)*m43-2*m32*(s-t+u)*m4+3*(s2-t2+u2)*m4-2*m3*(2*m44-(3*s+t+3*u)*m42+u*(t+u)))*m22+(-2*(s-t+u)*m44+3*(s2-t2+u2)*m42-2*m3*(4*m44-4*(s+t+u)*m42+s2+(t+u)**2+2*s*t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(-2*(s-t+u)*m42+s2-t2+u2-4*s*u))*m2+8*m_X4*((m3-m4)*m22-(m32-4*m4*m3+m42-s)*m2-m32*m4+m3*(m42-t)+m4*u)+m4*(2*(t-u)*m44+(s2-2*t*s-3*t2+3*u2)*m42+2*m3*s*(m42-s-t)*m4-s3+t3-u3+s*t2+s*u2-t*u2-s2*t+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u))+4*m_X2*(m25+(m4-m3)*m24+2*(m32+m42-t-u)*m23+(6*m4*m32-(4*m42+s-t+u)*m3+m4*(2*m42-s-3*t))*m22+(m44-(3*t+u)*m42-2*m3*(s+u)*m4-s2+t2+u2+m32*(6*m42+s-2*t)-s*u+t*u)*m2+m32*m4*(2*m42-2*t+u)+m4*(m44-2*(s+t)*m42+s2+t2-u2+s*(t-u))+m3*(-m44-(s-t+u)*m42+t*u+s*(t+2*u))))*m1+8*m_X4*((m32-m4*m3-t)*m22+(m4*m32+(u-m42)*m3-m4*t)*m2+m3*m4*s+m32*(m42-t)+t*(t-m42))-2*m_X2*(2*(m3-m4)*m25+(2*m3*m4+s-t-u)*m24-2*(m3-m4)*(2*m3*m4+s+t+u)*m23+(4*m3*m43-6*t*m42-2*m3*(s+2*u)*m4-s2+t2+u2+m32*(8*m42+s-t-3*u)+2*t*u)*m22-2*(2*m4*(m42-s-u)*m32-(m44-(2*s+u)*m42+s*(t+u))*m3+m4*(m44-(s+t+u)*m42+s*t+2*s*u+t*u))*m2+m4*(m4*(-3*s-t+u)*m32+2*(m42-s-t)*(m42-u)*m3-m4*(m42-s-t-u)*(s+t-u)))-m2*m4*(-2*(-2*m42+2*m3*m4+s-t)*m24+(m4*(2*m42-s+t-u)+m3*(2*u-4*m42))*m23+(4*m44+(-6*s+2*t-6*u)*m42+2*m3*(-2*m42+3*s+t+u)*m4+3*s2-3*t2+u2-2*t*u+2*m32*(-s+t+u))*m22-(m4*(s-t+u)*m32+2*(2*m44-(s+t+3*u)*m42+u*(t+u))*m3+m4*((s-t+u)*m42-s2+t2-u2))*m2-s3+t3-u3+m42*s2-3*m42*t2+s*t2+3*m42*u2+s*u2-t*u2+2*m3*m4*s*(m42-s-t)+2*m44*t-s2*t-2*m42*s*t-2*m44*u+s2*u+t2*u+2*s*t*u+m32*(2*(s+t-u)*m42+s2-t2+u2-4*s*u))) * s_prop*u_prop*((s-m_X2)*(u-m_X2) + m_Gamma_X2)

    tt = 1/m_X4*4*(-m18+(-2*m22-2*m42+2*s+t+2*u)*m16+(4*m_X2*m3-2*m3*t)*m15+(2*m34+2*m42*m32-2*s*m32+3*t*m32-2*u*m32-s2+t2-u2+2*m42*s+m42*t+2*m2*m4*t-2*s*t-2*m_X2*(4*m32+2*m2*m4-s+t-u)+2*m42*u-2*s*u-2*t*u+m22*(2*m32-4*m42+2*s+t+2*u))*m14+4*m3*(2*m_X2-t)*(m22+m32+m42-s-u)*m13+(2*(m22-2*m4*m2+2*m32+m42-s-u)*m_X4-2*(4*m34+(8*m42-2*(3*s+t+3*u))*m32+s2-t2+u2-m42*s+m42*t-2*m2*m4*(2*m32+t)+m22*(8*m32+4*m42-3*s+t-u)-3*m42*u+2*s*u)*m_X2-t3+2*m34*m42+2*m32*s2-2*m32*t2+m42*t2+2*m32*u2+t*u2-2*m34*s-4*m32*m42*s+3*m34*t+6*m32*m42*t+s2*t-4*m32*s*t-2*m42*s*t-2*m2*m4*t*(2*m32+t)+m22*(2*m34+(8*m42-4*s+6*t-4*u)*m32+t*(4*m42-2*s+t-2*u))-2*m34*u-4*m32*m42*u+4*m32*s*u-4*m32*t*u-2*m42*t*u+2*s*t*u)*m12-2*m3*(2*(m22-4*m4*m2+m42-t)*m_X4-2*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u)*m_X2+t*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u))*m1+2*m_X4*((m32+2*m42-s-u)*m22+2*m4*(t-m32)*m2+s2+u2-m42*s+m32*(m42-s-u)-m42*u)-m32*(m32-t)*(m34+2*(m42-s-u)*m32+s2-t2+u2-2*m42*s+m42*t-2*m2*m4*t+m22*(2*m32+4*m42-2*s+t-2*u)-2*m42*u+2*s*u)-2*m_X2*m32*((4*m42-s+t-3*u)*m22+2*m4*(m32-t)*m2+s2-t2+u2-3*m42*s+m42*t-m42*u+2*s*u-m32*(s-t+u))) * t_prop

    uu = 1/m_X4*4*(-m18+(-2*m22-2*m32+2*s+2*t+u)*m16+(4*m_X2*m4-2*m4*u)*m15+(2*m44+2*m32*m42-2*s*m42-2*t*m42+3*u*m42-s2-t2+u2+2*m32*s+2*m32*t-2*s*t+m32*u+2*m2*m3*u-2*s*u-2*t*u-2*m_X2*(4*m42+2*m2*m3-s-t+u)+m22*(-4*m32+2*m42+2*s+2*t+u))*m14+4*m4*(m22+m32+m42-s-t)*(2*m_X2-u)*m13+(2*(m22-2*m3*m2+m32+2*m42-s-t)*m_X4-2*(4*m44-6*s*m42-6*t*m42-2*u*m42+s2+t2-u2+2*s*t-2*m2*m3*(2*m42+u)+m32*(8*m42-s-3*t+u)+m22*(4*m32+8*m42-3*s-t+u))*m_X2+2*m32*m44-u3+2*m42*s2+2*m42*t2+m32*u2-2*m42*u2-2*m44*s-4*m32*m42*s-2*m44*t-4*m32*m42*t+4*m42*s*t+3*m44*u+6*m32*m42*u+s2*u+t2*u-2*m32*s*u-4*m42*s*u-2*m32*t*u-4*m42*t*u+2*s*t*u-2*m2*m3*u*(2*m42+u)+m22*(2*m44+(-4*s-4*t+6*u)*m42+4*m32*(2*m42+u)+u*(-2*s-2*t+u)))*m12-2*m4*(2*(m22-4*m3*m2+m32-u)*m_X4-2*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u))*m_X2+u*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u)))*m1+2*m_X4*((2*m32+m42-s-t)*m22+2*m3*(u-m42)*m2+s2+t2-m42*s+m32*(m42-s-t)-m42*t)-m42*(m42-u)*(m44-2*s*m42-2*t*m42+s2+t2-u2+2*s*t-2*m2*m3*u+m32*(2*m42-2*s-2*t+u)+m22*(4*m32+2*m42-2*s-2*t+u))-2*m_X2*m42*((4*m32-s-3*t+u)*m22+2*m3*(m42-u)*m2-(m42-s-t-u)*(s+t-u)+m32*(-3*s-t+u))) * u_prop

    tu = 1/m_X4*2*(2*m18+(4*m22+4*m32+4*m42-5*s-3*t-3*u)*m16+2*(-2*m33-2*m4*m32-2*m42*m3+s*m3+t*m3-2*m43+4*m_X2*m2+m4*s+m2*(-2*m32-2*m42+s)+m4*u)*m15+(-4*(2*m32+2*m4*m3+2*m42-s)*m_X2+m22*(4*m32+4*m42-5*s-3*t-3*u)+m2*(4*m33+4*m4*m32+4*m42*m3-2*(s+u)*m3+4*m43-2*m4*(s+t))+2*(m34+2*m4*m33+(2*m42-3*s-t-u)*m32+(2*m43-m4*s)*m3+m44+2*(s+t)*(s+u)-m42*(3*s+t+u)))*m14+(-4*m35-4*m4*m34-8*m42*m33+8*s*m33+4*t*m33+4*u*m33-8*m43*m32+8*m4*s*m32+4*m4*t*m32+4*m4*u*m32-4*m44*m3-3*s2*m3-t2*m3+u2*m3+8*m42*s*m3+4*m42*t*m3-4*s*t*m3+4*m42*u*m3-2*s*u*m3-4*t*u*m3-4*m45-3*m4*s2+m4*t2-m4*u2+8*m43*s+4*m43*t-2*m4*s*t+4*m43*u-4*m4*s*u-4*m4*t*u+4*m22*(-2*m33-2*m4*m32+(-2*m42+s+t)*m3+m4*(-2*m42+s+u))+4*m_X2*(2*m33+2*m4*m32+(2*m42-s+t-u)*m3+2*m2*(m32+m42-s-t-u)+m4*(2*m42-s-t+u))-2*m2*(2*m34+(4*m42-3*s-t-3*u)*m32+2*m44+s*(s+t+u)-m42*(3*s+3*t+u)))*m13+(4*m4*m35-s*m34+t*m34+u*m34+8*m43*m33-8*m4*s*m33-4*m4*t*m33-4*m4*u*m33+2*s2*m32-2*t2*m32-2*u2*m32-3*m42*s*m32+3*m42*t*m32+3*m42*u*m32+4*m45*m3+3*m4*s2*m3-m4*t2*m3-m4*u2*m3-8*m43*s*m3-4*m43*t*m3+2*m4*s*t*m3-4*m43*u*m3+2*m4*s*u*m3+4*m4*t*u*m3-s3+t3+u3+2*m42*s2-2*m42*t2+s*t2-2*m42*u2+s*u2-t*u2+8*m_X4*(m32+m4*m3+m42-m2*(m3+m4)-s)-m44*s+m44*t-s2*t+m44*u-s2*u-t2*u-2*s*t*u+m22*(8*m4*m33+(4*m42-s+t+u)*m32+(8*m43-4*m4*s)*m3+s2-t2-u2+4*t*u+m42*(-s+t+u))-2*m_X2*(4*m34+(16*m42-7*s-5*(t+u))*m32-4*m4*s*m3+4*m44+2*s2+2*m22*(6*m32+4*m4*m3+6*m42-3*s)-7*m42*s-5*m42*t+2*s*t-5*m42*u+2*s*u-2*m2*(4*m4*m32+(4*m42-s+t)*m3+m4*(u-s)))+2*m2*(2*m35+2*m4*m34+(4*m42-3*s-t-3*u)*m33+(4*m43-3*m4*(s+t+u))*m32+(2*m44-3*(s+t+u)*m42+s2+u2+s*t+2*s*u)*m3+m4*(2*m44-(3*s+3*t+u)*m42+s2+t2+2*s*t+s*u)))*m12+(4*m42*m35+2*s*m35-2*t*m35+4*m43*m34+2*m4*s*m34-2*m4*t*m34-2*m4*u*m34+4*m44*m33-3*s2*m33+3*t2*m33+u2*m33+2*m42*s*m33-6*m42*t*m33-6*m42*u*m33-2*s*u*m33+4*m45*m32-3*m4*s2*m32+3*m4*t2*m32+3*m4*u2*m32+2*m43*s*m32-6*m43*t*m32-6*m43*u*m32+s3*m3-t3*m3-u3*m3-3*m42*s2*m3+3*m42*t2*m3-s*t2*m3+3*m42*u2*m3-s*u2*m3+t*u2*m3+2*m44*s*m3-2*m44*t*m3+s2*t*m3-2*m44*u*m3+s2*u*m3+t2*u*m3+2*s*t*u*m3+m4*s3-m4*t3-m4*u3-3*m43*s2+m43*t2-m4*s*t2+3*m43*u2-m4*s*u2+m4*t*u2+2*m45*s+m4*s2*t-2*m43*s*t-2*m45*u+m4*s2*u+m4*t2*u+2*m4*s*t*u-8*m_X4*((m3+m4)*m22-(m32+4*m4*m3+m42-s)*m2+m32*m4+m3*(m42-t)-m4*u)+m22*(2*(s-t+u)*m33+2*m4*(s-t-u)*m32+(2*(s-t-u)*m42-s2+t2+u2-4*t*u)*m3+m4*(2*(s+t-u)*m42-s2+t2+u2-4*t*u))+4*m_X2*(m35+m4*m34+2*(m42-s-u)*m33+m4*(2*m42-3*s-t)*m32+(m44-(3*s+u)*m42+s2-t2+u2+s*u-t*u)*m3+m4*(m44-2*(s+t)*m42+s2+t2-u2+s*t-t*u)+m22*(2*m33+6*m4*m32+(6*m42-2*s+t)*m3+m4*(2*m42-2*s+u))-m2*(m34+(4*m42-s+t+u)*m32+2*m4*(t+u)*m3+m44-s*t-s*u-2*t*u+m42*(-s+t+u)))-2*m2*(4*m4*m35+(2*m42-u)*m34+4*m4*(2*m42-s-t-u)*m33+(2*m44-(s+3*(t+u))*m42+u*(s+u))*m32+m4*(4*m44-4*(s+t+u)*m42+s2+t2+u2+2*s*(t+u))*m3+m42*t*(-m42+s+t)))*m1+8*m_X4*((m32+m4*m3+m42-s)*m22+(-m4*m32+(u-m42)*m3+m4*t)*m2+s*(-m32-m4*m3-m42+s))+2*m_X2*(2*m4*m35+(s-t+u)*m34-2*m4*(s+t+u)*m33+(6*s*m42-s2+t2-u2-2*s*u)*m32+2*m4*(m44-(s+t+u)*m42+2*t*u+s*(t+u))*m3+m42*(m42-s-t-u)*(s+t-u)+m22*(4*m4*m33+(-8*m42+s-t+3*u)*m32+4*m4*(m42-t-u)*m3+m42*(s+3*t-u))-2*m2*(m35+m4*m34+(2*m42-s-t-u)*m33+m4*(2*m42-t-2*u)*m32+(m44-(2*t+u)*m42+t*(s+u))*m3+m4*(m42-s-t)*(m42-u)))-m3*m4*(2*(2*m42+s-t)*m34+m4*(2*m42+s-t-u)*m33+(4*m44+2*(s-3*(t+u))*m42-3*s2+3*t2+u2-2*s*u)*m32+m4*((s-t-u)*m42-s2+t2+u2)*m3+s3-t3-u3-3*m42*s2+m42*t2-s*t2+3*m42*u2-s*u2+t*u2+2*m44*s+s2*t-2*m42*s*t-2*m44*u+s2*u+t2*u+2*s*t*u+m22*(2*(s-t+u)*m32+m4*(s-t-u)*m3-s2+t2+u2+2*m42*(s+t-u)-4*t*u)-2*m2*(2*m4*m34+(2*m42-u)*m33+m4*(2*m42-s-3*t-u)*m32+(2*m44-(s+t+3*u)*m42+u*(s+u))*m3+m4*t*(-m42+s+t)))) * t_prop*u_prop*((t-m_X2)*(u-m_X2)+m_Gamma_X2)

    # Anton: Think returning only s-channel would be sufficient
    # return vert*ss
    # return vert*(ss + tt + uu + st + su + tu)
    # Anton: Relevant process using this function has no s-channel 
    return vert*(tt + uu + tu)

@nb.jit(nopython=True, cache=True)
def M2_gen_ss(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False):
    m12 = m1*m1
    m13 = m1*m12
    m14 = m12*m12
    m15 = m12*m13
    m16 = m13*m13
    m18 = m14*m14

    m22 = m2*m2
    m24 = m22*m22
    m26 = m22*m24
    m28 = m24*m24

    m32 = m1*m1
    m42 = m1*m1
    m_X4 = m_X2*m_X2

    u = m12 + m22 + m32 + m42 - s - t
    s2 = s*s
    t2 = t*t
    u2 = u*u

    s_prop = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)

    ss = 1/m_X4*4*(-m18+(-2*m32-2*m42+s+2*t+2*u)*m16+(4*m_X2*m2-2*m2*s)*m15+(2*m24+(2*m32+2*m42+3*s-2*t-2*u)*m22-4*m32*m42+s2-t2-u2+m32*s+m42*s+2*m3*m4*s+2*m32*t+2*m42*t-2*s*t-2*m_X2*(4*m22+2*m3*m4+s-t-u)+2*m32*u+2*m42*u-2*s*u-2*t*u)*m14+4*m2*(2*m_X2-s)*(m22+m32+m42-t-u)*m13+(2*(2*m22+m32+m42-2*m3*m4-t-u)*m_X4-2*(4*m24+(8*m32-4*m4*m3+8*m42-2*(s+3*(t+u)))*m22-s2+t2+u2+m42*s-2*m3*m4*s-m42*t+m32*(4*m42+s-3*t-u)-3*m42*u+2*t*u)*m_X2+m24*(2*m32+2*m42+3*s-2*t-2*u)+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u)))+2*m22*((4*m42+3*s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2-2*s*t-2*s*u+2*t*u+m42*(3*s-2*(t+u))))*m12-2*m2*(2*(m32-4*m4*m3+m42-s)*m_X4-2*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u)))*m_X2+s*(m24+2*(m32+m42-t-u)*m22-s2+t2+u2+m42*s-2*m3*m4*s-2*m42*t-2*m42*u+2*t*u+m32*(4*m42+s-2*(t+u))))*m1-m28+m26*(-2*m32-2*m42+s+2*t+2*u)+2*m_X4*((2*m42-t-u)*m32+2*m4*s*m3+t2+u2-m42*(t+u))+m24*(-2*(2*m3*m4+s-t-u)*m_X2+s2-t2-u2+m42*s+2*m3*m4*s+2*m42*t-2*s*t+2*m42*u-2*s*u-2*t*u+m32*(-4*m42+s+2*(t+u)))+m22*(2*(m32-2*m4*m3+m42-t-u)*m_X4-2*((4*m42+s-t-3*u)*m32-2*m4*s*m3-s2+t2+u2+m42*(s-3*t-u)+2*t*u)*m_X2+s*((4*m42+s-2*(t+u))*m32-2*m4*s*m3-s2+t2+u2+2*t*u+m42*(s-2*(t+u))))) * s_prop*(s_prop*((s-m_X2)*(s-m_X2) - m_Gamma_X2) if sub else 1.)

    return vert*ss

@nb.jit(nopython=True, cache=True)
def M2_fi(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    aa --> dd
    """
    m_d4 = m_d2*m_d2
    m_d6 = m_d2*m_d4
    m_d8 = m_d4*m_d4
    m_X4 = m_X2*m_X2
    u = 3.*m_d2 - s - t
    s2 = s*s
    t2 = t*t
    s_prop = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)

    ss = 8*(2*(m_d4-2*m_d2*t+t*(s+t))+s2) * s_prop
    tt = (4*(2*m_X4*(m_d4-2*m_d2*(2*s+t)+2*s2+2*s*t+t2)+4*m_X2*m_d4*s+m_d4*(m_d2-t)**2))/m_X4 * t_prop
    uu = (4*(2*m_X4*(m_d4-2*m_d2*(s+t)+s2+t2)+4*m_X2*m_d4*s+m_d4*(-m_d2+s+t)**2))/m_X4 * u_prop
    # Anton:|M|^2 contains cross-term Re(D_BW(s)* x D_BW(t)) = ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    st = (8*(2*m_X2*(m_d4-m_d2*(s+2*t)+(s+t)**2)+m_d2*(m_d4+m_d2*(s-2*t)+t2)))/m_X2 * ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    su = -((8*(2*m_X2*(m_d4+m_d2*(s-2*t)+t2)+m_d2*(m_d4-m_d2*(s+2*t)+(s+t)**2)))/m_X2) * ((s-m_X2)*(u-m_X2) + m_Gamma_X2)*s_prop*u_prop
    tu = -((4*(4*m_X4*s*(3*m_d2-s)+2*m_X2*m_d2*(2*m_d4-4*m_d2*(s+t)+s2+2*s*t+2*t2)+m_d8+m_d6*(s-2*t)+m_d4*t*(s+t)))/m_X4) * ((t-m_X2)*(u-m_X2) + m_Gamma_X2)*t_prop*u_prop

    # return vert*ss
    return vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_tr(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    ad --> dd
    """
    m_d4 = m_d2*m_d2
    u = 3.*m_d2 - s - t
    s2 = s*s
    t2 = t*t
    s_prop = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)

    ss = 8*(2*m_d4-m_d2*(s+6*t)+s2+2*s*t+2*t2) * s_prop
    tt = 8*(2*m_d4-m_d2*(6*s+t)+2*s2+2*s*t+t2) * t_prop
    uu = 8*(8*m_d4-5*m_d2*(s+t)+s2+t2) * u_prop
    # Anton:|M|^2 contains cross-term Re(D_BW(s)* x D_BW(t)) = ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    st = -16*(2*m_d2-s-t)*(m_d2+s+t) * ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    su = -16*(4*m_d4-5*m_d2*t+t2) * ((s-m_X2)*(u-m_X2) + m_Gamma_X2)*s_prop*u_prop
    tu = 16*(4*m_d4-5*m_d2*s+s2) * ((t-m_X2)*(u-m_X2) + m_Gamma_X2)*t_prop*u_prop

    # return vert*ss
    return vert*(ss + tt + uu + st + su + tu)

@nb.jit(nopython=True, cache=True)
def M2_el(s, t, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    dd --> dd
    """
    m_d4 = m_d2*m_d2
    u = 3.*m_d2 - s - t
    s2 = s*s
    t2 = t*t
    s_prop = 1. / ((s-m_X2)*(s-m_X2) + m_Gamma_X2)
    t_prop = 1. / ((t-m_X2)*(t-m_X2) + m_Gamma_X2)
    u_prop = 1. / ((u-m_X2)*(u-m_X2) + m_Gamma_X2)
    
    ss = 8*(8*m_d4-8*m_d2*t+s2+2*t*(s+t)) * s_prop
    tt = 8*(2*(s-2*m_d2)**2+2*s*t+t2) * t_prop
    uu = 8*(24*m_d4-8*m_d2*(s+t)+s2+t2) * u_prop
    # Anton:|M|^2 contains cross-term Re(D_BW(s)* x D_BW(t)) = ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    st = -16*(4*m_d4-(s+t)**2) * ((s-m_X2)*(t-m_X2) + m_Gamma_X2)*s_prop*t_prop
    su = -16*(12*m_d4-8*m_d2*t+t2) * ((s-m_X2)*(u-m_X2) + m_Gamma_X2)*s_prop*u_prop
    tu = 16*(12*m_d4-8*m_d2*s+s2) * ((t-m_X2)*(u-m_X2) + m_Gamma_X2)*t_prop*u_prop

    # return vert*ss
    return vert*(ss + tt + uu + st + su + tu)

########################################################
# Anton: Cross-sections for each process gen, tr, fi, el

@nb.jit(nopython=True, cache=True)
def ker_sigma_gen_new(t, s, p1cm, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub):
    # Anton: Numerical issues with integration (some numbers get extremely large for sigma_el). 
    # Trick: Scale integrand by R and re-scale result back with 1/R
    # return 1e-3*M2_gen_new_3(s, t, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub)/(64.*np.pi*s*p1cm*p1cm)
    return M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False)/(64.*np.pi*s*p1cm*p1cm)

# no factor taking care of identical particles (not known on this level)
# @nb.jit(nopython=True, cache=True)
def sigma_gen_new(s, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub=False):
    """
    Anton: 
    Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we try to integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 --> CM: sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2 in CM-frame
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3^2 + m2/4^2)]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space: H(E_cm - (m3 + m4)), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1^2 + m2^2)]^2 - 4*m1^2*m2^2}) = H([s - (m1^2 + m2^2)]^2 - 4*m1^2*m2^2)
    => [s - (m1^2 + m2^2)]^2 > 4*m1^2*m2^2 => s - (m1^2 + m2^2) > 2*m1*m2, s - (m1^2 + m2^2) < -2*m1*m2
    => s > (m1 + m2)^2, s < (m1 - m2)^2, latter never satisfied -- omit last solution 
    = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    g1*g2*sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*E_cm^2*p1cm^2) * int_{t_lower}^{t_upper} dt |M|^2
    g1, g2 spin factor from initial dof. in |M|^2. We use g1*g2*sigma and not sigma, as we use these for the 
    thermally averaged cross-section
    Note: This function can not be vectorized using 'quad' or 'quad_vec' as boundaries also will be arrays. 
          Use np.vectorize(sigma_gen)(s, m1, ...) instead if array output is wanted.
    """
    # Anton: Area where the three-momenta is defined, heavyside-functions - H(s-(m1+m2)^2)*H(s-(m3+m4)^2)
    if s < (m1 + m2)**2. or s < (m3 + m4)**2.:
        return 0.

    # Anton: Make upper and lower integration bounds 
    E1cm = (s + m1*m1 - m2*m2) / (2*np.sqrt(s))
    E3cm = (s + m3*m3 - m4*m4) / (2*np.sqrt(s))
    p1cm = np.sqrt((E1cm - m1)*(E1cm + m1))
    p3cm = np.sqrt((E3cm - m3)*(E3cm + m3))

    E13diff = (m1*m1 - m2*m2 - m3*m3 + m4*m4) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm))
    t_lower = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm))
    M2_t_integrate, err = quad(ker_sigma_gen_new, t_lower, t_upper, args=(s, p1cm, m1, m2, m3, m4, vert, m_d2, m_X2, m_h2, m_Gamma_X2, m_Gamma_h2, sub))

    # Anton: No symmetry-factors, as this is unkown at this level
    return 1e3*M2_t_integrate

@nb.jit(nopython=True, cache=True)
def ker_sigma_gen(t, s, p1cm, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub):
    # Anton: Numerical issues with integration (some numbers get extremely large for sigma_el). 
    # Trick: Scale integrand by R and re-scale result back with 1/R
    return 1e-3*M2_gen(s, t, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub)/(64.*np.pi*s*p1cm*p1cm)

# no factor taking care of identical particles (not known on this level)
# @nb.jit(nopython=True, cache=True)
def sigma_gen(s, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub=False):
    """
    Anton: 
    Since sigma ~ int d(cos(theta)) |M|^2 for 2 to 2 process, we try to integrate |M|^2 analytically. 
    Switch integration to t = m_d^2 + m_phi^2 - 2E1*E3 + 2p1*p3*cos(theta), d(cos(theta)) = 1/(2*p1*p3)dt
    Since sigma is Lorentz invariant, calculate in CM-frame
    t = (p1-p3)^2 = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm - E3cm)^2 - (p1cm^2 + p3cm^2 - 2*p1cm*p3cm*cos(theta))
    This gives upper and lower bounds (cos(theta)=1, cos(theta)=-1)
    t_upper = (E1cm - E3cm)^2 - (p1cm - p3cm)^2 = (E1cm-E3cm + (p1cm-p3cm))*(E1cm-E3cm - (p1cm-p3cm))
    t_lower = (E1cm - E3cm)^2 - (p1cm + p3cm)^2 = (E1cm-E3cm + (p1cm+p3cm))*(E1cm-E3cm - (p1cm+p3cm))
    s = (p1/3 + p2/4)^2 = (E1/3cm + E2/4cm)^2 --> CM: sqrt(s) = E1/3cm + E2/4cm
    Trick: E2/4^2 = E1/3^2 - m1/3^2 + m2/4^2 in CM-frame
    => (sqrt(s) - E1/3cm)^2 = E1/3cm^2 - m1/3^2 + m2/4^2
    => E1/3cm = (s + m1/3^2 - m2/4^2) / (2*sqrt(s))
    which would also give momentum 
    p1/3cm = sqrt(E1/3cm^2 - m1/3^2) = 1/(2*sqrt(s))*sqrt([s - (m1/3^2 + m2/4^2)]^2 - 4*m1/3^2*m2/4^2)
    for integration bounds. 
    Two heavysides - one from integration of phase-space: H(E_cm - (m3 + m4)), one from demanding p1/2cm positive: 
    H(1/(4*s)*{[s - (m1^2 + m2^2)]^2 - 4*m1^2*m2^2}) = H([s - (m1^2 + m2^2)]^2 - 4*m1^2*m2^2)
    => [s - (m1^2 + m2^2)]^2 > 4*m1^2*m2^2 => s - (m1^2 + m2^2) > 2*m1*m2, s - (m1^2 + m2^2) < -2*m1*m2
    => s > (m1 + m2)^2, s < (m1 - m2)^2, latter never satisfied -- omit last solution 
    = H(s - (m1 + m2)^2) = H(E_cm - m1 - m2)
    Cross-section:
    g1*g2*sigma = H(E_cm - m3 - m4)*H(E_cm - m1 - m2)/(64*pi*E_cm^2*p1cm^2) * int_{t_lower}^{t_upper} dt |M|^2
    g1, g2 spin factor from initial dof. in |M|^2. We use g1*g2*sigma and not sigma, as we use these for the 
    thermally averaged cross-section
    Note: This function can not be vectorized using 'quad' or 'quad_vec' as boundaries also will be arrays. 
          Use np.vectorize(sigma_gen)(s, m1, ...) instead if array output is wanted.
    """
    # Anton: Area where the three-momenta is defined, heavyside-functions - H(s-(m1+m2)^2)*H(s-(m3+m4)^2)
    if s < (m1 + m2)**2. or s < (m3 + m4)**2.:
        return 0.

    # Anton: Make upper and lower integration bounds 
    E1cm = (s + m1*m1 - m2*m2) / (2*np.sqrt(s))
    E3cm = (s + m3*m3 - m4*m4) / (2*np.sqrt(s))
    p1cm = np.sqrt((E1cm - m1)*(E1cm + m1))
    p3cm = np.sqrt((E3cm - m3)*(E3cm + m3))

    E13diff = (m1*m1 - m2*m2 - m3*m3 + m4*m4) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm))
    t_lower = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm))

    M2_t_integrate, err = quad(ker_sigma_gen, t_lower, t_upper, args=(s, p1cm, m1, m2, m3, m4, vert, m_X2, m_Gamma_X2, sub))

    # Anton: No symmetry-factors, as this is unkown at this level
    return 1e3*M2_t_integrate

@nb.jit(nopython=True, cache=True)
def ker_sigma_fi(t, s, p1cm, m_d2, vert, m_X2, m_Gamma_X2):
    return M2_fi(s, t, m_d2, vert, m_X2, m_Gamma_X2)/(64.*np.pi*s*p1cm*p1cm)


# @nb.jit(nopython=True, cache=True)
def sigma_fi(s, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    aa -> dd
    """
    # Anton: Heavyside function H(s)*H(s - 4*m_d2)
    if s < 0 or s < 4*m_d2:
        return 0.
    
    m_d = np.sqrt(m_d2)

    # Anton: Make upper and lower integration bounds 
    E1cm = s / (2*np.sqrt(s))
    E3cm = s / (2*np.sqrt(s))
    p1cm = E1cm
    p3cm = np.sqrt((E3cm - m_d)*(E3cm + m_d))

    t_upper = -(p1cm - p3cm)**2
    t_lower = -(p1cm + p3cm)**2
    
    M2_t_integrate, err = quad(ker_sigma_fi, t_lower, t_upper, args=(s, p1cm, m_d2, vert, m_X2, m_Gamma_X2))

    # factor 0.5 due to identical particles in final state
    return M2_t_integrate / 2


@nb.jit(nopython=True, cache=True)
def ker_sigma_tr(t, s, p1cm, m_d2, vert, m_X2, m_Gamma_X2):
    return M2_tr(s, t, m_d2, vert, m_X2, m_Gamma_X2)/(64.*np.pi*s*p1cm*p1cm)

# @nb.jit(nopython=True, cache=True)
def sigma_tr(s, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    ad -> dd
    """
    # Anton: Heavyside function H(s - m_d2)*H(s - 4*m_d2)
    if s < m_d2 or s < 4*m_d2:
        return 0.

    m_d = np.sqrt(m_d2)

    # Make upper and lower integration bounds 
    E1cm = (s + m_d*m_d) / (2*np.sqrt(s))
    E3cm = s / (2*np.sqrt(s))
    p1cm = np.sqrt((E1cm - m_d)*(E1cm + m_d))
    p3cm = np.sqrt((E3cm - m_d)*(E3cm + m_d))

    E13diff = (m_d*m_d) / (2*np.sqrt(s))
    t_upper = (E13diff + (p1cm - p3cm))*(E13diff - (p1cm - p3cm))
    t_lower = (E13diff + (p1cm + p3cm))*(E13diff - (p1cm + p3cm))
    
    M2_t_integrate, err = quad(ker_sigma_tr, t_lower, t_upper, args=(s, p1cm, m_d2, vert, m_X2, m_Gamma_X2))

    # factor 0.5 due to identical particles in final state
    return M2_t_integrate / 2


@nb.jit(nopython=True, cache=True)
def ker_sigma_el(t, s, p1cm, m_d2, vert, m_X2, m_Gamma_X2):
    # Anton: Numerical issues with integration (some numbers get extremely large). 
    # Trick: Scale integrand by R and re-scale result back with 1/R
    return 1e-3*M2_el(s, t, m_d2, vert, m_X2, m_Gamma_X2)/(64.*np.pi*s*p1cm*p1cm)

# d d --> d d 
# @nb.jit(nopython=True, cache=True)
def sigma_el(s, m_d2, vert, m_X2, m_Gamma_X2):
    """
    Anton:
    dd -> dd
    """
    # Anton: Heavyside function
    if s < 4.*m_d2:
        return 0.

    m_d = np.sqrt(m_d2)

    # Anton: Make upper and lower integration bounds 
    E1cm = np.sqrt(s) / 2
    E3cm = np.sqrt(s) / 2
    p1cm = np.sqrt((E1cm - m_d)*(E1cm + m_d))
    p3cm = np.sqrt((E3cm - m_d)*(E3cm + m_d))

    # t_upper = -(p1cm - p3cm)**2 = 0 automatically
    # t_lower = -4*p1cm**2
    t_upper = 0
    t_lower = -4*p1cm**2

    M2_t_integrate, err = M2_t_integrate_1, err = quad(ker_sigma_el, t_lower, t_upper, args=(s, p1cm, m_d2, vert, m_X2, m_Gamma_X2))
    # factor 0.5 due to identical particles in final state
    return 1e3*M2_t_integrate / 2

if __name__ == '__main__':
    import matplotlib.pyplot as plt 

    m_ratio = 3

    m_d = 1e-5      # GeV. 1e-5 GeV = 10 keV
    m_a = 0.
    m_X = 5*m_d
    m_h = 3*m_d
    sin2_2th = 1e-12
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    y = 2e-4

    # Anton: fi = aa->dd, tr = ad->dd, el = ss->ss
    # Anton: For small theta << 1, sin^2(theta) = 1/4*sin^2(2*theta)
    vert_fi = y**4 * np.cos(th)**4*np.sin(th)**4
    vert_tr = y**4 * np.cos(th)**6*np.sin(th)**2
    vert_el = y**4 * np.cos(th)**8

    print(f'vert_fi: {vert_fi:.2e}, vert_tr: {vert_tr:.2e}, vert_el: {vert_el:.2e}')

    m_d2 = m_d*m_d
    m_X2 = m_X*m_X
    m_h2 = m_h*m_h

    th_arr = np.linspace(0, 2*np.pi, 1000)
    GammaX = Gamma_X(y=y, th=th_arr, m_X=m_X, m_d=m_d)
    GammaX_new = Gamma_X_new(y=y, th=th_arr, m_X=m_X, m_d=m_d)
    GammaPhi = Gamma_phi(y=y, th=th_arr, m_phi=m_h, m_d=m_d, m_X=m_X)
    plt.plot(th_arr, GammaX, 'r')
    plt.plot(th_arr, GammaX_new, 'tab:green')
    plt.plot(th_arr, GammaPhi, 'tab:blue')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], labels=[r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$', r'$2\pi$'])
    plt.show()

    m1 = m_a
    m2 = m_d
    s_min = (m1 + m2)**2
    # S = np.linspace(s_min, 1e2, int(1e3))
    S = 10**(np.linspace(np.log10(s_min), 2, int(1e3)))
    T = np.linspace(-1, 1, int(1e3))
    # T = np.concatenate((-10**(np.linspace(1, 0, int(1e3/2))), 10**(np.linspace(0, 1, int(1e3/2)))))
    s, t = np.meshgrid(S, T, indexing='ij')

    GammaX = Gamma_X(y=y, th=th, m_X=m_X, m_d=m_d)
    GammaX_new = Gamma_X_new(y=y, th=th, m_X=m_X, m_d=m_d)
    GammaPhi = Gamma_phi(y=y, th=th, m_phi=m_h, m_d=m_d, m_X=m_X)
    m_Gamma_X2 = (m_X*GammaX)**2
    m_Gamma_X2_new = (m_X*GammaX_new)**2
    m_Gamma_h2 = (m_h*GammaPhi)**2
    print(GammaX, GammaX_new)

    s_sigma = 10**(np.linspace(np.log10(3*m_d2), 2, int(1e3)))
    # Transmission ad --> dd
    time1 = time.time()
    sigma_gen_tr = np.vectorize(sigma_gen)(s=s_sigma, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, sub=True)
    time2 = time.time()
    print(time2-time1)
    sigma_gen_new_tr = np.vectorize(sigma_gen_new)(s=s_sigma, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2_new, m_Gamma_h2=m_Gamma_h2, sub=True)
    time3 = time.time()
    print(time3-time2)
     # Freeze-in aa --> dd
    sigma_gen_fi = np.vectorize(sigma_gen)(s=s_sigma, m1=m_a, m2=m_a, m3=m_d, m4=m_d, vert=vert_fi, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, sub=True)
    sigma_gen_new_fi = np.vectorize(sigma_gen_new)(s=s_sigma, m1=m_a, m2=m_a, m3=m_d, m4=m_d, vert=vert_fi, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2_new, m_Gamma_h2=m_Gamma_h2, sub=True)
     # Elastic dd --> dd
    sigma_gen_el = np.vectorize(sigma_gen)(s=s_sigma, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_X2=m_X2, m_Gamma_X2=m_Gamma_X2, sub=True)
    sigma_gen_new_el = np.vectorize(sigma_gen_new)(s=s_sigma, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2_new, m_Gamma_h2=m_Gamma_h2, sub=True)


    # sigma_trans = np.vectorize(sigma_tr)(s=s_sigma, m_d2=m_d**2, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    # # Elastic dd --> dd
    # sigma_gen_el = np.vectorize(sigma_gen)(s=s_sigma, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2, sub=False)
    # sigma_elast = np.vectorize(sigma_el)(s=s_sigma, m_d2=m_d**2, vert=vert_el, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    # # Freeze-in aa --> dd
    # sigma_gen_fi = np.vectorize(sigma_gen)(s=s_sigma, m1=m_a, m2=m_a, m3=m_d, m4=m_d, vert=vert_fi, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2, sub=False)
    # sigma_freezein = np.vectorize(sigma_fi)(s=s_sigma, m_d2=m_d**2, vert=vert_fi, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,6))

    ax1.loglog(s_sigma, sigma_gen_tr, 'k--', label='sigma_gen|tr')
    ax1.loglog(s_sigma, sigma_gen_new_tr, 'r--', label='sigma_gen_new|tr')
    ax2.loglog(s_sigma, sigma_gen_fi, 'k--', label='sigma_gen|fi')
    ax2.loglog(s_sigma, sigma_gen_new_fi, 'r--', label='sigma_gen_new|fi')
    ax3.loglog(s_sigma, sigma_gen_el, 'k--', label='sigma_gen|el')
    ax3.loglog(s_sigma, sigma_gen_new_el, 'r--', label='sigma_gen_new|el')

    ax1.axvline(m_h2, ls='--')
    ax2.axvline(m_h2, ls='--')
    ax3.axvline(m_h2, ls='--')
    ax1.axvline(m_X2, ls='-.')
    ax2.axvline(m_X2, ls='-.')
    ax3.axvline(m_X2, ls='-.')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    # ax2.legend()

    fig.tight_layout()
    plt.show()

    # fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(10,6))
    
    # ax1.loglog(s_sigma, 2*sigma_trans, 'r', label='sigma_tr', lw=2.5)
    # ax1.loglog(s_sigma, sigma_gen_tr, 'k--', label='sigma_gen|tr')
    # ax1.loglog(s_sigma, np.max(2*sigma_trans)*s_min/s_sigma, 'grey', linestyle='--', zorder=-1)

    # ax2.loglog(s_sigma, 2*sigma_elast, 'r', label='sigma_el', lw=2.5)
    # ax2.loglog(s_sigma, sigma_gen_el, 'k--', label='sigma_gen|el')
    # ax2.loglog(s_sigma, np.max(2*sigma_elast)*s_min/s_sigma, 'grey', linestyle='--', zorder=-1)

    # ax3.loglog(s_sigma, 2*sigma_freezein, 'r', label='sigma_fi', lw=2.5)
    # ax3.loglog(s_sigma, sigma_gen_fi, 'k--', label='sigma_gen|fi')
    # ax3.loglog(s_sigma, np.max(2*sigma_freezein)*s_min/s_sigma, 'grey', linestyle='--', zorder=-1)

    # ax1.axvline(4*m_d**2, color='grey', linestyle='-.')
    # ax2.axvline(4*m_d**2, color='grey', linestyle='-.')
    # # ax3.axvline(4*m_d**2, color='grey', linestyle='-.')
    # ax1.legend()
    # ax2.legend()
    # ax3.legend()
    # ax3.set_xlabel('s [GeV]', fontsize=16, weight='bold')
    # # ax1.plot(s_sigma, 1e-23*np.ones_like(s_sigma), marker='.', linestyle='none')
    # plt.savefig('sterile_test/cross_sections_tr_el_fi.pdf')
    # plt.show()

    # 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    # M2_trans = np.vectorize(M2_tr)(s, t, m_d2=m_d**2, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    # M2_general_tHoft = M2_gen_tHoft(s, t, m1=m_a, m2=m_d, m3=m_d, m4=m_d, vert=vert_tr, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    # M2_elast = np.vectorize(M2_el)(s, t, m_d2=m_d**2, vert=vert_el, m_X2=m_X**2, m_Gamma_X2=m_Gamma_X2)
    M2_general_false = M2_gen_new_3(s=s, t=t, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, sub=False)
    plot_M2 = ax1.contourf(s, t, np.log10(M2_general_false), levels=300, cmap='jet')
    fig1.colorbar(plot_M2)
    ax1.set_xscale('log')
    ax1.plot(m_X**2, 0, 'ko')

    # 2 
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.plot(m_X**2, 0, 'ko')

    M2_general_true = M2_gen_new_3(s=s, t=t, m1=m_d, m2=m_d, m3=m_d, m4=m_d, vert=vert_el, m_d2=m_d2, m_X2=m_X2, m_h2=m_h2, m_Gamma_X2=m_Gamma_X2, m_Gamma_h2=m_Gamma_h2, sub=True)
    # Anton: They are equal up to ~ r decimals, where abs((a-b)/b) = C*10^(-r)
    # print(-np.log10(np.max(abs((M2_general_tHoft - M2_general)/M2_general))))
    plot_M2_gen = ax2.contourf(s, t, np.log10(M2_general_true), levels=300, cmap='jet')
    cbar2 = fig2.colorbar(plot_M2_gen)
    # plt.plot(S[::10], np.zeros_like(S[::10]), 'ko')
    ax2.set_xscale('log')

    # Anton: Can make colorbar interactive - see constant value that you click
    from matplotlib import colors
    highlight_cmap = colors.ListedColormap(['k'])
    highlight = ax2.imshow(np.ma.masked_all_like(np.log10(M2_general_true)), interpolation='nearest', vmin=np.log10(M2_general_true).min(), vmax=np.log10(M2_general_true).max(), extent=[S.min(),S.max(),T.min(), T.max()], cmap=highlight_cmap, origin='lower', aspect='auto', zorder=10)

    # highlight = [ax2.contour(s, t, (M2_general_true), colors='none')]

    def on_pick(event):
        val = event.mouseevent.ydata
        selection = np.ma.masked_outside(np.log10(M2_general_true), val-0.2, val+0.2)
        highlight.set_data(selection.T)
        # highlight[0].remove()
        # highlight[0] = ax2.contour(s, t, selection, colors='k')
        fig2.canvas.draw()
    cbar2.ax.set_picker(5)
    fig2.canvas.mpl_connect('pick_event', on_pick)

    fig1.tight_layout()
    fig2.tight_layout()
    plt.show()