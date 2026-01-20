# PandemicRevival
Sterile_caller.py
- Only one function: `call(...)`. Run this file if you want a single simulation of PDM. Is used in `find_y.py`.
- Uses function `Gamma_X_new(...)` from `vector_mediator.py` to get total decay rate for X, and classes `TimeTempRelation` to relate SM temperatur and time, and `Pandemolator` for solving Boltzmann equation from `pandemolator.py`.

 C_res_vector.py
- Main purpose: Set up total collision operator in `sterile_caller.py`.
- Contains relevant collision operators for number density, energy density involving the vector boson.
- Contains cross-sections and thermally averaged cross-sections

C_res_vector_no_spin_stat.py
- The spin-statistics-less cousin of C_res_vector, making simplifications for when spin-statistics does not matter. Is not fully updated, is used in `C_therm_kd(...)` in `sterile_caller.py`. 

vector_mediator.py
- Contains `Gamma_X_new(...)` and more generally matrix elements, and some cross-sections. Most of these are not called (I think only `M2_gen(...)` and `sigma_gen_new(...)` are the only functions ever called in addition). 

find_y.py
- Makes parameter scan of `sin2_2theta, m_d` while adjusting correct coupling strength `y` to match the DM relic density. Finds `y` by using some sort of root-finding method, calling `sterile_caller.call` to check if given `y` matches relic.