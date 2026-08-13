import scipy as sp

def test_function(x, n):
    return x**n

def test_quad():
    sol = sp.integrate.quad(test_function, 0, 1, args=(2,))
    print(sol)
    return 0

test_quad()