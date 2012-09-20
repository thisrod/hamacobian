# Using a fixed trust region, see how well a single coherent state coverges to its own expansion

from plot import plotens
from naive import jsq, jlft, jfock, rho
from numpy import array, zeros, identity, log, sum, exp, sqrt, linspace
from numpy.linalg import solve as gesv
from scipy.misc import factorial
from levenmarq import *
	
alpha = 2+0j
n = array(xrange(20))
# the truncation error has norm 1e-4 
original = exp(-0.5*abs(alpha)**2) * alpha**n / sqrt(factorial(n))
original /= sqrt(sum(original**2))
# coefficients of the halfway state expanded over Fock states
halfway = (-1)**(n*(n-1)/2) * original

set_target(original)
print "Fitting a coherent state to the expansion of |2+0j>"
print "eps  alpha  norm  residual   (ten steps)"
for epsilon in [0.115, 0.12, 0.15, 0.2, 0.3, 0.5, 1, 2]:
	zs = array([[0, 1.8]])
	zs[:,0] -= 0.5*lnormsq(zs)
	for i in xrange(10):
		zs, epsilon = lmstep(zs, epsilon)
	print "%.3f   %.2f    %.2f    %.0e" % (epsilon, zs[0,1], exp(0.5*lnormsq(zs)), sqrt(residual(zs)))
	
set_target(halfway)
print
print "Fitting two coherent states to the corresponding halfway state"
print "eps  norm  residual   (ten steps)"
for epsilon in [0.493, 0.495, 0.5, 1, 1.5, 2, 5, 50, 100]:
	zs = array([[0, 1.8j], [0, -1.8j]])
	zs[:,0] -= 0.5*lnormsq(zs)
	for i in xrange(10):
		zs, epsilon = lmstep(zs, epsilon)
	print "%.3f    %.2f    %.0e" % (epsilon, exp(0.5*lnormsq(zs)), sqrt(residual(zs)))

print
print "Fitting four coherent states to the halfway state"
print "eps  norm  residual   (ten steps)"
for epsilon in [1.185, 1.2, 1.25, 1.3, 1.5, 2, 5, 50, 100]:
	zs = array([[0, 0.2+2j], [0, -0.2+2j], [0, 0.2-2j], [0, -0.2-2j]])
	zs[:,0] -= 0.5*lnormsq(zs)
	for i in xrange(10):
		zs, epsilon = lmstep(zs, epsilon)
	print "%.3f    %.2f    %.0e" % (epsilon, exp(0.5*lnormsq(zs)), sqrt(residual(zs)))

