# math
from math import log, pow
# numpy
import numpy as np
# numba
import numba as nb


class LogInterp(object):

    def __init__(self, x_grid, y_grid, base=np.e, extrap=None):
        self._sBase    = base
        self._sLogBase = log(self._sBase)

        self._sExtrap = extrap

        self._sXLog = np.log(x_grid)/self._sLogBase
        self._sYLog = np.log(y_grid)/self._sLogBase

        self._sXminLog = self._sXLog[ 0]
        self._sXmaxLog = self._sXLog[-1]
        if self._sXmaxLog <= self._sXminLog:
            raise ValueError(
                "The values in x_grid need to be in ascending order."
            )

        self._sYminLog = self._sYLog[ 0]
        self._sYmaxLog = self._sYLog[-1]

        self._sN = len(self._sXLog)

        self._sCache = {}


    def _apply_extrap_strategy(self, x_log):
        split = self._sExtrap.split(':')

        # Extract the id: (b, c, p)
        id = split[0]
        if id == 'c':
            extrap_vals = split[1].split(',')

            extrap_val_min = float( extrap_vals[0] )
            extrap_val_max = float( extrap_vals[1] )

            # When this function is called, x_log is garantueed to be out-of-bounds
            return ( extrap_val_min if x_log < self._sXminLog else extrap_val_max )

        if id == 'b':
            extrap_val_min = pow( self._sBase, self._sYminLog )
            extrap_val_max = pow( self._sBase, self._sYmaxLog )

            # When this function is called, x_log is garantueed to be out-of-bounds
            return ( extrap_val_min if x_log < self._sXminLog else extrap_val_max )

        if id == 'p':
            extrap_exps = split[1].split(',')

            extrap_exp_min = float( extrap_exps[0] )
            extrap_exp_max = float( extrap_exps[1] )

            # TODO

        raise ValueError(
            "The given value for 'extrap' is not a valid identifier."
        )


    def _perform_interp(self, x):
        x_log = log(x)/self._sLogBase

        if not (self._sXminLog <= x_log <= self._sXmaxLog):
            if self._sExtrap is None:
                raise ValueError(
                    "The given value does not lie within the interpolation range."
                )

            return self._apply_extrap_strategy(x_log)

        ix = int( ( x_log - self._sXminLog )*( self._sN - 1 )/( self._sXmaxLog - self._sXminLog ) )

        # Handle the case for which ix+1 is out-of-bounds
        if ix == self._sN - 1: ix -= 1

        x1_log, x2_log = self._sXLog[ix], self._sXLog[ix+1]
        y1_log, y2_log = self._sYLog[ix], self._sYLog[ix+1]

        m = ( y2_log - y1_log )/( x2_log - x1_log )
        b = y2_log - m*x2_log

        return pow( self._sBase, m*x_log + b )


    def __call__(self, x):
        if x not in self._sCache:
            self._sCache[x] = self._perform_interp(x)

        return self._sCache[x]

# 2d-interpolation (bilinear) on equally spaced grid (same grid for both dimensions)
@nb.jit(nopython=True, cache=True)
def interp_2d(x1, x2, x_grid, y_grid):
    if x1 < x_grid[0] or x1 > x_grid[-1] or x2 < x_grid[0] or x2 > x_grid[-1]:
        return np.nan

    n = x_grid.size
    delta_x = (x_grid[-1]-x_grid[0])/(n-1)

    i1 = min(int((x1 - x_grid[0])/delta_x), n-2)
    i2 = min(int((x2 - x_grid[0])/delta_x), n-2)

    d1 = (x1 - x_grid[i1])/delta_x
    d2 = (x2 - x_grid[i2])/delta_x

    r0 = y_grid[i1, i2  ]*(1.-d1)+y_grid[i1+1, i2  ]*d1
    r1 = y_grid[i1, i2+1]*(1.-d1)+y_grid[i1+1, i2+1]*d1

    return r0*(1.-d2)+r1*d2

# 3d-interpolation (trilinear) on equally spaced grid (same grid for all dimensions)
@nb.jit(nopython=True, cache=True)
def interp_3d(x1, x2, x3, x_grid, y_grid):
    if x1 < x_grid[0] or x1 > x_grid[-1] or x2 < x_grid[0] or x2 > x_grid[-1] or x3 < x_grid[0] or x3 > x_grid[-1]:
        return np.nan

    n = x_grid.size
    delta_x = (x_grid[-1]-x_grid[0])/(n-1)

    i1 = min(int((x1 - x_grid[0])/delta_x), n-2)
    i2 = min(int((x2 - x_grid[0])/delta_x), n-2)
    i3 = min(int((x3 - x_grid[0])/delta_x), n-2)

    d1 = (x1 - x_grid[i1])/delta_x
    d2 = (x2 - x_grid[i2])/delta_x
    d3 = (x3 - x_grid[i3])/delta_x

    r00 = y_grid[i1, i2  , i3  ]*(1.-d1)+y_grid[i1+1, i2  , i3  ]*d1
    r01 = y_grid[i1, i2  , i3+1]*(1.-d1)+y_grid[i1+1, i2  , i3+1]*d1
    r10 = y_grid[i1, i2+1, i3  ]*(1.-d1)+y_grid[i1+1, i2+1, i3  ]*d1
    r11 = y_grid[i1, i2+1, i3+1]*(1.-d1)+y_grid[i1+1, i2+1, i3+1]*d1

    r0 = r00*(1.-d2)+r10*d2
    r1 = r01*(1.-d2)+r11*d2

    return r0*(1.-d3)+r1*d3

# Cummulative numerical Simpson integration
def cumsimp(x_grid, y_grid):
    n = len(x_grid)

    # Integration is performed in log-space
    delta_z = log( x_grid[-1]/x_grid[0] )/( n-1 )
    g_grid  = x_grid*y_grid

    i_grid = np.zeros( n )

    last_even_int = 0.
    for i in range(1, n//2 + 1):
        ie = 2 * i
        io = 2 * i - 1

        i_grid[io] = last_even_int + 0.5 * delta_z * (g_grid[io-1] + g_grid[io])
        if ie < n:
            i_grid[ie] = last_even_int + delta_z * (g_grid[ie-2] + 4.*g_grid[ie-1] + g_grid[ie])/3.
            last_even_int = i_grid[ie]

    return i_grid

# Cummulative numerical Simpson integration
@nb.jit(nopython=True, cache=True)
def simp(x_grid, y_grid):
    n = x_grid.size
    if n <= 1:
        return 0.

    # Integration is performed in log-space
    delta_z = log( x_grid[-1]/x_grid[0] )/( n-1 )
    g_grid  = delta_z*x_grid*y_grid

    # Multiply with weights
    if n == 2:
        g_grid *= 0.5
    elif n//2 == 1:
        g_grid[::2] *= 2./3.
        g_grid[0] *= 0.5
        g_grid[-1] *= 0.5
        g_grid[1::2] *= 4./3.
    else:
        g_grid[0] *= 1./3.
        g_grid[1:-1:2] *= 4./3.
        g_grid[2:-2:2] *= 2./3.
        g_grid[-2] *= 5./6.
        g_grid[-1] *= 0.5

    return np.sum(g_grid)

# Computes Simpson weights (see above)
def simp_wgts(n):
    wgts = np.ones(n)
    if n == 2:
        wgts *= 0.5
    elif n//2 == 1:
        wgts[::2] *= 2./3.
        wgts[0] *= 0.5
        wgts[-1] *= 0.5
        wgts[1::2] *= 4./3.
    else:
        wgts[0] *= 1./3.
        wgts[1:-1:2] *= 4./3.
        wgts[2:-2:2] *= 2./3.
        wgts[-2] *= 5./6.
        wgts[-1] *= 0.5

    return wgts

# Cummulative numerical Simpson integration for vectorial integrand
# integration over second index of y
@nb.jit(nopython=True, cache=True)
def simp_vec(x_grid, y_grid):
    n = x_grid.size
    if n <= 1:
        return np.zeros(len(y_grid))

    # Integration is performed in log-space
    delta_z = log( x_grid[-1]/x_grid[0] )/( n-1 )
    g_grid  = delta_z*x_grid*y_grid

    # Multiply with weights
    if n == 2:
        g_grid *= 0.5
    elif n//2 == 1:
        g_grid[:,::2] *= 2./3.
        g_grid[:,0] *= 0.5
        g_grid[:,-1] *= 0.5
        g_grid[:,1::2] *= 4./3.
    else:
        g_grid[:,0] *= 1./3.
        g_grid[:,1:-1:2] *= 4./3.
        g_grid[:,2:-2:2] *= 2./3.
        g_grid[:,-2] *= 5./6.
        g_grid[:,-1] *= 0.5

    return np.sum(g_grid, axis=1)

# Five-point stencil numerical differentiation
def fpsder(x_grid, y_grid):
    n = len(x_grid)

    # Differentiation is performed in log-space
    h_grid = np.log( x_grid )

    d_grid = np.zeros( n )

    # First two entries
    d_grid[0] = ( y_grid[1] - y_grid[0] )/( h_grid[1] - h_grid[0] )
    d_grid[1] = ( y_grid[2] - y_grid[0] )/( h_grid[2] - h_grid[0] )

    # Entries in between
    for i in range(2, n - 2):
        nom = -y_grid[i+2]/12. + 2.*y_grid[i+1]/3. - 2.*y_grid[i-1]/3. + y_grid[i-2]/12.
        den = h_grid[i+1] - h_grid[i]

        d_grid[i] = nom/den

    # Last two entries
    l = n - 1
    d_grid[l-1] = ( y_grid[l] - y_grid[l-2] )/( h_grid[l] - h_grid[l-2] )
    d_grid[l  ] = ( y_grid[l] - y_grid[l-1] )/( h_grid[l] - h_grid[l-1] )

    return d_grid/x_grid


# Moving average
def mavg(x_grid, n=2):
    # Check if n is a positive integer
    if not abs( int(n) ) == n:
        raise ValueError('The parameter n must be a positive integer or zero')

    # Treat the special case 'n=0'
    if n == 0:
        return x_grid

    cx_grid = np.cumsum( np.insert(x_grid, 0, 0.) )

    # Calculate the averaged grids
    ax_grid = ( cx_grid[n:] - cx_grid[:-n] ) / float(n)

    # Append the first and last element
    ax_grid = np.array( [ x_grid[0], *ax_grid, x_grid[-1] ] )

    return ax_grid
