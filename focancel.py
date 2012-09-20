# How closely can superpositions of coherent states fit the cat state halfway between quartic oscillator revivials?

from plot import plotens
from naive import jsq, jlft, jfock, rho
from numpy import array, zeros, identity, log, sum, exp, sqrt, linspace
from numpy.linalg import solve as gesv
from scipy.misc import factorial
	
alpha = 2+0j
n = array(xrange(20))
# the truncation error has norm 1e-4 
original = exp(-0.5*abs(alpha)**2) * alpha**n / sqrt(factorial(n))
original /= sqrt(sum(original**2))
# coefficients of the halfway state expanded over Fock states
halfway = (-1)**(n*(n-1)/2) * original

target = original

def lnormsq(z):
	return log(sum(rho(z,z))).real
	
def residual(z):
	"squared Hilbert space distance between the ensemble z and the target state"
	bkts = jfock(z, target)
	return 1 + exp(lnormsq(z)) - 2*sum(bkts[::2]).real

def tikstep(A, y, epsilon):
	"The Tikhonov solution to Ax=y, with regularisation parameter epsilon"
	A = A.copy() + epsilon*identity(y.size)
	return gesv(A, y)
	
def lmstep(z, epsilon):
	"update the guess or the confidence interval.  in this version, we keep epsilon fixed and return the linear guess."
	r = rho(z,z)
	V = jsq(z, r)
	rhs = jfock(z, target) - jlft(z, r)
	return z+tikstep(V, rhs, epsilon).reshape(z.shape), epsilon
		
zs = array([[0, 1.8]])
zs[:,0] -= 0.5*lnormsq(zs)
print "alpha  norm  residual"
print "%.3f    %.3g    %.3g" % (zs[0,1], exp(0.5*lnormsq(zs)), sqrt(residual(zs)))epsilon = 0.1
for i in xrange(10):
	zs, epsilon = lmstep(zs, epsilon)
	print "%.3f    %.3g    %.3g" % (zs[0,1], exp(0.5*lnormsq(zs)), sqrt(residual(zs)))

# plotens(zs)
