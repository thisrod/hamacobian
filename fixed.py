# Using a fixed trust region, see how well a single coherent state coverges to its own expansion

from plot import plotens
from naive import jsq, jlft, jfock, rho
from numpy import array, zeros, identity, log, sum, exp, sqrt, linspace
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
	
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yscale('log')
plt.title("Convergence using Levenberg-Marquardt with fixed trust region")
plt.xlabel("Reciprocal trust region, epsilon")
plt.ylabel("Residual after 10 steps")
	
set_target(original)
A = tabulate([[0, 1.8]], [0.12, 0.15, 0.2, 0.3, 0.5, 1, 2], 10)
set_target(halfway)
B = tabulate([[0, 1.8j], [0, -1.8j]], [0.493, 0.495, 0.5, 1, 1.5, 1.7, 2.5], 10)
C= tabulate([[0, 0.2+2j], [0, -0.2+2j], [0, 0.2-2j], [0, -0.2-2j]],
			 [1.185, 1.2, 1.25, 1.3, 1.5, 2, 3], 10)

ax.plot(A.keys(), A.values(), "o", label = "1 ket to |2+0j>")
ax.plot(B.keys(), B.values(), "*", label = "2 kets to cat")
ax.plot(C.keys(), C.values(), "D", label = "4 kets to cat")
plt.legend()
plt.show()
