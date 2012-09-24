# Using a fixed trust region, see how well a single coherent state coverges to its own expansion

from plot import plotens
from naive import jsq, jlft, jfock, rho
from numpy import array, zeros, identity, log, sum, exp, sqrt, linspace, pi
from numpy.linalg import solve as gesv
from scipy.misc import factorial
from levenmarq import *
import matplotlib.pyplot as plt
	
alpha = 2+0j
n = array(xrange(20))
# the truncation error has norm 1e-4 
original = exp(-0.5*abs(alpha)**2) * alpha**n / sqrt(factorial(n))
original /= sqrt(sum(original**2))
# coefficients of the halfway state expanded over Fock states
halfway = (-1)**(n*(n-1)/2) * original

def tabulate(z, epsilons, steps):
	results = {}
	for eps in epsilons:
		zs = array(z)
		zs[:,0] -= 0.5*lnormsq(zs)
		for i in xrange(steps):
			zs, whatever = lmstep(zs, eps)
		results[eps] = sqrt(residual(zs))
	return results
	
def rg(*args):
	return list(linspace(*args))
	
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
plt.title("Convergence using Levenberg-Marquardt with fixed trust region")
plt.xlabel("Reciprocal trust region, epsilon")
plt.ylabel("Residual after 10 steps")
	
set_target(original)
A = tabulate([[0, 1.8]], rg(0.12, 5, 15) + rg(0.15, 0.4, 15), 10)
set_target(halfway)
B = tabulate([[0, 1.8j], [0, -1.8j]], [0.495, 0.5,0.7,  1, 1.4, 1.7, 2.5], 10)
C= tabulate([[0, 0.2+2.05j], [0, -0.2+2j], [0, 0.2-2.05j], [0, -0.2-2j]],
			 [1.185, 1.2, 1.25, 1.3, 1.5, 2, 3], 10)
four = [ [0, 1j**n] for n in xrange(4) ]
D= tabulate(four, rg(0.3, 0.6, 5) + rg(0.7,5,15), 10)
six = [ [0, exp(pi/3*n*1j)] for n in xrange(6) ]
E= tabulate(six, [0.4, 0.6, 1.0, 1.5, 2, 3], 10)

zs = array(four)
zs[:,0] -= 0.5*lnormsq(zs)
plotens(zs)
for i in xrange(6):
	zs, whatever = lmstep(zs, 0.4)
	plotens(zs)

zs = array(six)
zs[:,0] -= 0.5*lnormsq(zs)
plotens(zs)
for i in xrange(10):
	zs, whatever = lmstep(zs, 1.0)
	plotens(zs)

ax.plot(A.keys(), A.values(), "o", label = "1 ket to |2+0j>")
ax.plot(B.keys(), B.values(), "v", label = "2 kets to cat")
ax.plot(C.keys(), C.values(), "D", label = "4 tight samples")
ax.plot(D.keys(), D.values(), "s", label = "4 roots of unity")
ax.plot(E.keys(), E.values(), "s", label = "6 roots of unity")
plt.legend()
plt.show()
