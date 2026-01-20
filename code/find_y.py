#! /usr/bin/env python3

import numpy as np
from multiprocessing import Pool, cpu_count
from pebble import ProcessPool, ProcessExpired
from concurrent.futures import TimeoutError
import time
import sys, os

import constants_functions as cf
import sterile_caller

"""
Anton:
Finds coupling y that satisfies Omega*h^2 = 0.12 by
using midpoint method (I think) to pick y and solve 
with sterile-caller --> pandemolator. 

Error-message:
something went wrong...  Unable to allocate 1.16 MiB for an array with shape (38014, 4) and data type float64 <class 'numpy.core._exceptions._ArrayMemoryError'> find_y.py 73
something went wrong...  Allocation failed (probably too large). <class 'MemoryError'> find_y.py 73
"""

# Anton: Mass ratio m_X / m_d
r_m_Xd = 2.5
# Anton: SM neutrino mass set to zero
m_nu = 0.

# Anton: Need m0 >> mi, m12 >> mi^2 / m0, i = a,1,2,(12)
m0 = 1e3
m12 = 1e-1
m2 = 1e-2
ma = m12*1e-2

# Anton: Fermion (Fermi) = 1, Boson = 1 (Bose)
k_d = 1.
k_X = -1.   
k_nu= 1.        

dof_d = 2.      # Anton: Fermion 2 spin dof.
dof_X = 3.      # Anton: Massive vector boson 3 polarization dof.

n_m = 10         # 21
n_th = 10        # 81
# Anton: Search for keV-scale sterile neutrinos. Input is in GeV, so (m_d keV) = (1e-6*m_d GeV)
m_d_grid = 1e-6*np.logspace(0, 2.5, n_m)    # 10^a keV - 10^b keV, a=0, b=2.5
sin2_2th_grid = np.logspace(-18, -8, n_th)


num_cpus = cpu_count()
num_process = int(0.25*num_cpus) # cpu_count(), 48

params_grid = np.array((np.repeat(m_d_grid, n_th), np.tile(sin2_2th_grid, n_m))).T

# Anton: If spin-statistics should be included (True) or not (False)
spin_facs = True
# Anton: If intermediate particle are off-shell (True) or on-shell (False)
off_shell = False

dirname = './sterile_res/'
i_max = 60      # 60
i_skip = 20
def find_y(params):
    m_d = params[0]
    m_N1 = m_d
    m_N2 = m_d

    sin2_2th = params[1]
    th = 0.5*np.arcsin(np.sqrt(sin2_2th))
    m_X = r_m_Xd*m_d 

    O_d_h2_dw = cf.O_h2_dw(m_d, th)         # Anton: Omega_DM * h^2 from Dodelson-Widrow mechanism
    log_enhance_req = np.log(cf.omega_d0/O_d_h2_dw)
    if log_enhance_req < 0.:
        print(f'find_y.py returned m_d: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y: {np.nan:.2e}')
        return m_d, m_X, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    y_cur = np.sqrt(log_enhance_req/(1e19*sin2_2th/m_X)) # 1.65e16 roughly the scaling factor from parameter scans
    print(f'm_d: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y_cur: {y_cur:.2e}, O_d_h2_dw: {O_d_h2_dw:.2e}, log_enhance_req: {log_enhance_req:.2e}')
    O_d_h2_old = O_d_h2_dw
    y_old = 0.
    max_y = 1.
    min_y = 0.
    md_percent = np.log10(m_d/m_d_grid[0])/np.log10(m_d_grid[-1]/m_d_grid[0])
    sin22th_percent = np.log10(sin2_2th/sin2_2th_grid[0])/np.log10(sin2_2th_grid[-1]/sin2_2th_grid[0])
    print(f'Starting {md_percent*100}% of m_d, {sin22th_percent*100}% of sin2_2th')
    for i in range(i_max + 1):
        print(f'Iteration {i} of {i_max}, {i/i_max*100:.1f}%, md: {m_d:.2e}({md_percent*100:.1f}%), sin22th: {sin2_2th:.2e}({sin22th_percent*100:.1f}%), total {(i/i_max+md_percent+sin22th_percent)/3*100:.1f}%')
        try:
            # time1 = time.time()
            # print("Running sterile_caller.call ")

            t_grid, T_SM_grid, T_nu_grid, ent_grid, hubble_grid, sf_grid, T_d_grid, xi_d_grid, xi_X_grid, n_d_grid, n_X_grid, C_therm_grid, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3, reached_integration_end = sterile_caller.call(m_N1, m_N2, m_X, m0, m12, m2, ma, k_d, k_X, k_nu, dof_d, dof_X, y_cur, spin_facs=True, off_shell=False)

            # print(f"sterile_caller.call ran in {time.time()-time1}s")

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("something went wrong in find_y.py... ", e, exc_type, fname, exc_tb.tb_lineno)
            print(f'find_y.py returned {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y: {np.nan:.2e}')
            return m_d, m_X, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        O_d_h2_cur = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
        print(f'Use m_d: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y_cur: {y_cur:.2e}, O_d_g2_cur: {O_d_h2_cur:.2e}')
        if O_d_h2_cur > cf.omega_d0:
            max_y = min(y_cur, max_y)
        elif O_d_h2_cur < cf.omega_d0 and ((y_cur > y_old) == (O_d_h2_cur > O_d_h2_old) or i == 0):
            min_y = max(y_cur, min_y)
        if np.abs(O_d_h2_cur-cf.omega_d0)/cf.omega_d0 < 1e-2 or np.abs(max_y-min_y)/max_y < 1e-6:
            break
        elif ((y_cur > y_old) != (O_d_h2_cur > O_d_h2_old) and i > 0) or not reached_integration_end: # already in freeze-out regime, reduce y_cur to go back to pandemic
            y_old = y_cur
            O_d_h2_old = O_d_h2_cur
            max_y = min(y_cur, max_y)
            y_cur = 0.5*(min_y + max_y)
            print('freeze-out regime or very large abundance, reducing y', m_d, sin2_2th, O_d_h2_dw, y_old, y_cur)
        elif i < i_max:
            log_enhance_cur = np.log(O_d_h2_cur/O_d_h2_dw)
            y_old = y_cur
            O_d_h2_old = O_d_h2_cur
            y_cur = min(np.sqrt(log_enhance_req/log_enhance_cur), 2.)*y_cur if log_enhance_cur > 0. else 2.*y_cur
            if y_cur >= max_y:
                y_cur = 0.5*(y_old + max_y)
            elif y_cur <= min_y:
                y_cur = 0.5*(y_old + min_y)
            if np.abs(y_old-y_cur)/y_old < 1e-6:
                y_cur = y_old
                break
            # print(m_d, sin2_2th, O_d_h2_dw, log_enhance_req, log_enhance_cur, y_old, y_cur)

    print('find_y.py for-loop done ')
    md_str = f'{m_d:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_d:.5e}'.split('e')[1]
    mX_str = f'{m_X:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{m_X:.5e}'.split('e')[1]
    sin22th_str = f'{sin2_2th:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{sin2_2th:.5e}'.split('e')[1]
    y_str = f'{y_cur:.5e}'.split('e')[0].rstrip('0').rstrip('.') + 'e' + f'{y_cur:.5e}'.split('e')[1]
    filename = f'md_{md_str};mX_{mX_str};sin22th_{sin22th_str};y_{y_str};full.dat'
    # Anton: Save benchmark-points
    # np.savetxt(dirname+'benchmark_pts_mX3_mh3/'+filename, np.column_stack((t_grid[::-i_skip][::-1], T_SM_grid[::-i_skip][::-1], T_nu_grid[::-i_skip][::-1], ent_grid[::-i_skip][::-1], hubble_grid[::-i_skip][::-1], sf_grid[::-i_skip][::-1], T_d_grid[::-i_skip][::-1], xi_d_grid[::-i_skip][::-1], xi_X_grid[::-i_skip][::-1], xi_h_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1], n_X_grid[::-i_skip][::-1], n_h_grid[::-i_skip][::-1], C_therm_grid[::-i_skip][::-1], n_d_grid[::-i_skip][::-1]*m_d*cf.s0/(ent_grid[::-i_skip][::-1]*cf.rho_crit0_h2))))

    therm_ratio = C_therm_grid / (3.*hubble_grid*n_d_grid)
    x_therm = m_d/T_nu_grid[np.argmax(therm_ratio > 1.)]
    x_d_therm = m_d/T_d_grid[np.argmax(therm_ratio > 1.)]
    therm_ratio_max = np.amax(therm_ratio)
    O_d_h2 = n_d_grid[-1]*m_d*cf.s0/(ent_grid[-1]*cf.rho_crit0_h2)
    print(f'find_y.py returned {m_d:.2e}, sin2_2th: {sin2_2th:.2e}, y: {y_cur:.2e}')
    print(f'Saved {filename} to benchmark_pts')
    return m_d, m_X, sin2_2th, y_cur, O_d_h2, x_therm, x_d_therm, therm_ratio_max, fs_length, fs_length_3, T_kd, T_kd_3, T_d_kd, T_d_kd_3, r_sound, r_sound_3

if __name__ == '__main__':
    time1 = time.time()
    print("Start process of find_y.py...")
    print(f"Number of CPUs: {num_cpus}")
    print(f"Number of processes: {num_process}")
    # with Pool(processes=num_process) as pool:
    #     # Anton: Returns 17 parameter values 
    #     results = pool.map(find_y, params_grid, chunksize=1)

    # Anton: Allows for timing out processes spending too much time 
    # Anton: days + hours + minutes [seconds]
    days = 0
    hours = 12
    minutes = 0
    # Anton: Set timeout. For no timeout, set value to None
    # timeout = days*24*60*60 + hours*60*60 + minutes*60
    timeout = None
    print(f"Timeout set to {timeout} seconds")
    with ProcessPool(max_workers=num_process) as pool:
        # Anton: Returns 17 parameter values 
        future = pool.map(find_y, params_grid, chunksize=1, timeout=timeout)

        iterator = future.result()

        result_list = []

        filename_params = dirname+f'rmXd_{r_m_Xd:.2e}_y_relic_test_{n_m}x{n_th}x{i_max}.dat'

        openfile = open(filename_params, "w")
        i_md = 0 
        i_sin22th = 0 
        while True:
            openfile = open(filename_params, "a")
            try:
                result = next(iterator)
                result_list.append(result)
                openfile.write(str(result).strip('()').replace(',', '')+'\n')
            except TimeoutError as error:
                m_d = m_d_grid[i_md]
                sin2_2th = sin2_2th_grid[i_sin22th]
                m_X = r_m_Xd*m_d 
                result = (m_d, m_X, sin2_2th, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                result_list.append(result)
                openfile.write(str(result).strip('()').replace(',', '')+'\n')
                print(f"function took longer than {timeout} seconds for md: {m_d:.2e}, sin2_2th: {sin2_2th:.2e}")
            except StopIteration:
                break
            except ProcessExpired as error:
                print("%s. Exit code: %d" % (error, error.exitcode))
            except Exception as error:
                print("function raised %s" % error)
                print(error.traceback)  # Python's traceback of remote process
            finally:
                i_sin22th += 1
                if i_sin22th == n_th:
                    i_sin22th = 0 
                    i_md += 1
                openfile.close()

        results = np.array(result_list).reshape(len(result_list), 17)

    dt = time.time() - time1
    print(f"find_y.py ran in {dt//60//60}h {dt//60%60}m {dt%60}s")
    # Anton: Save plots for plot of parameter-space
    print(dirname+f'rmXd_{r_m_Xd:.2e}_y_relic_test_{n_m}x{n_th}x{i_max}.dat')
    np.savetxt(dirname+f'rmXd_{r_m_Xd:.2e}_y_relic_test_{n_m}x{n_th}x{i_max}.dat', results)
