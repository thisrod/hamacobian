# How closely can a superposition of two coherent states fit the cat state halfway between quartic oscillator revivials?  Print the overlap, for amplitudes of the cat state close to 2.

from plot import plotens
from naive import jlft, jfock, rho
from numpy import array, zeros, log, sum, exp, sqrt, linspace
from scipy.misc import factorial

def lnorm(z):
	return 0.5*log(sum(rho(z,z)))

alpha = 2+0j
n = array(xrange(15))
cat = (-1)**(n*(n-1)/2) * exp(-0.5*abs(alpha)**2) * alpha**n / sqrt(factorial(n))

for x in linspace(1.8, 2.2, 10):
	zs = array([[0, x*1j], [0, -x*1j]])
	zs[:,0] -= lnorm(zs)
	bkts = jfock(zs, cat)
	print x, bkts[0,0] + bkts[2,0]

# plotens(z)
