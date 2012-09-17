# Numerical experiments in how badly the terms of Q cancel

from plot import plotens
from naive import jlft, jfock, rho
from numpy import array, zeros, log, sum


def lnorm(z):
	return 0.5*log(sum(rho(z,z)))

z = array([[0, 1.8], [0, 2.2]])
for n in xrange(z.shape[0]):
	z[n,0] = -lnorm(z[n:n+1,:])
z[:,0] -= lnorm(z)

print z

# plotens(z)

four = zeros(5);  four[4] = 1;

print jlft(z, rho(z,z))
print jfock(z, four)  # |4>
