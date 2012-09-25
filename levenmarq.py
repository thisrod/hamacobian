# Functions for Levenberg-Marquardt optimisation

# See states.py for the interface that x and z must implement
from numpy import array, identity, log, sum, exp
from numpy.linalg import solve as gesv

def set_target(x):
	global target
	target = x

def residual(z):
	"squared Hilbert space distance between the z and the target state"
	return target.norm()**2 + z.norm()**2 - 2*(z*target).real

def tikstep(A, y, epsilon):
	"The Tikhonov solution to Ax=y, with regularisation parameter epsilon"
	return gesv(A + epsilon*identity(y.size), y)
	
def step(z, epsilon):
	"update the guess or the confidence interval.  in this version, we keep epsilon fixed and return the linear guess."
	Dz = z.D()
	return z + tikstep(Dz*Dz, Dz*target - Dz*z, epsilon), epsilon
