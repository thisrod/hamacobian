# Numerical experiments in how badly the terms of Q cancel

from plot import plotens
from naive import jlft, rho
from numpy import array, log, sum


def lnorm(z):
	return 0.5*log(sum(rho(z,z)))

z = array([[0, 1.8], [0, 2.2]])
for n in xrange(z.shape[0]):
	z[n,0] = -lnorm(z[n:n+1,:])
z[:,0] -= lnorm(z)

# plotens(z)

print jlft(z, rho(z,z))
