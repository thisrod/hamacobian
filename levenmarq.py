# Functions for Levenberg-Marquardt optimisation

from moments import jn, jsq, jlft, rho
from numpy import array, matrix, zeros, identity, log, sum, exp, sqrt, linspace
from numpy.linalg import solve as gesv

target = None

def set_target(x):
	global target
	target = matrix(x.reshape((x.size, 1)))

def lnormsq(z):
	return log(sum(rho(z,z))).real

def residual(z):
	"squared Hilbert space distance between the ensemble z and the target state"
	bkts = jn(z, target.size) * target
	return 1 + exp(lnormsq(z)) - 2*sum(bkts[:,::2]).real

def tikstep(A, y, epsilon):
	"The Tikhonov solution to Ax=y, with regularisation parameter epsilon"
	A = A.copy() + epsilon*identity(y.size)
	return gesv(A, y)
	
def lmstep(z, epsilon):
	"update the guess or the confidence interval.  in this version, we keep epsilon fixed and return the linear guess."
	r = rho(z,z)
	V = jsq(z, r)
	rhs = jn(z, target.size) * target - jlft(z, r)
	return z+array(tikstep(V, rhs, epsilon)).reshape(z.shape), epsilon
		
