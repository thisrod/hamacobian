# Functions for Levenberg-Marquardt optimisation

from states import lccs, NState
from numpy import array, identity, log, sum, exp
from numpy.linalg import solve as gesv

target = None		# NState to approximate

def set_target(x):
	global target
	target = x

def residual(z):
	"squared Hilbert space distance between the ensemble z and the target state"
	return 1 + z.norm()**2 - 2*sum(z*target).real

def tikstep(A, y, epsilon):
	"The Tikhonov solution to Ax=y, with regularisation parameter epsilon"
	A = A.copy() + epsilon*identity(y.size)
	return gesv(A, y)
	
def lmstep(z, epsilon):
	"update the guess or the confidence interval.  in this version, we keep epsilon fixed and return the linear guess."
	j = z.D()
	V = j*j
	rhs = sum(j*target) - sum(j*z)
	return z+array(tikstep(V, rhs, epsilon)).reshape(z.shape), epsilon
		
