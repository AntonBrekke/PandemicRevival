using Interpolations

"""
Interpolation helpers for the z = log(x) parameterization, x = m_N1/T_nu.

`temp_interpolation` (utils.jl) builds log-log interpolants keyed on the
*decreasing* T_nu grid, and therefore has to reverse it first. The
background grids are monotonic in time, so z = log(x) = log(m_N1/T_nu) is
*increasing* and needs no reversal -- and is already the interpolation
coordinate, so a lookup costs one `exp` rather than `exp` -> divide ->
`log` -> `exp`.
"""

"""
    z_interpolation(z_grid, u; neg=false)

Log-log linear interpolation of `u` against an increasing `z_grid`. Set
`neg=true` for quantities that are strictly negative (e.g. dT_nu/dt) and so
cannot be log-interpolated directly -- mirrors the `neg` option of
`temp_interpolation` in utils.jl.
"""
function z_interpolation(z_grid, u; neg=false)
    if neg
        u = -u
    end
    log_u = log.(u)
    log_interp = linear_interpolation(z_grid, log_u, extrapolation_bc=Line())
    if neg
        return (z) -> -exp(log_interp(z))
    else
        return (z) -> exp(log_interp(z))
    end
end

"""
    x_grid_of(m_ref, T_nu_grid)

x = m_ref / T_nu on the background grid (increasing). Used to build
z_grid = log.(x_grid_of(...)) and as an intermediate for `dx_dt_grid_of`.
"""
x_grid_of(m_ref, T_nu_grid) = m_ref ./ T_nu_grid

"""
    dx_dt_grid_of(m_ref, T_nu_grid, dT_nu_dt_grid)

dx/dt = d(m_ref/T)/dt = -(x^2/m_ref) dT/dt, tabulated on the background
grid. Since dT_nu/dt < 0 this is strictly positive, so it can be
log-interpolated directly (no `neg` needed) -- unlike dT_nu/dt itself.
Storing dx/dt rather than dT_nu/dt also removes the per-evaluation
conversion done by `dx_dt_interp` in the T-based solver.
"""
function dx_dt_grid_of(m_ref, T_nu_grid, dT_nu_dt_grid)
    x_grid = x_grid_of(m_ref, T_nu_grid)
    return -(x_grid .^ 2) .* dT_nu_dt_grid ./ m_ref
end
