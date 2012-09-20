#
# Compute the inner product form, the squared Jacobian, and the Hamiltonian moments in the obvious way, using interpreted for loops.
#
# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.

from numpy import array, matrix, empty, sqrt, exp, sum, vander
from scipy.misc import factorial

	
def extract(z):
	# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.
	return z.shape + (z[:,0:1], z[:,1:])
	
def rho(z, w):
	"""Calculate the matrix of inner products between the components of two states."""
	nl, ml, fl, Al = extract(matrix(z))
	nr, mr, fr, Ar = extract(matrix(w))
	assert nl == nr and ml == mr
	logrho = fl.H + fr + Ar*Al.H
	return exp(logrho)

def jlft(z, rho):
	"""Calculate the product of the Jacobian bra matrix with the state.  Limited to one mode."""
	n, m, f, a = extract(matrix(z))
	Q = empty((2*n,1))
	Q[0:2*n:2,:] = sum(rho, 1)
	Q[1:2*n:2,:] = rho*a
	return Q
	
def jsq(z, rho):
	"Nothing"
	n, m, f, A = extract(matrix(z))
	w = matrix(z)
	w[:,0] = 1
	w = w.reshape(-1,1)
	poly = w*w.H
	for q in xrange(1,m):
		poly[q:n:n*m,q:n:n*m] += 1
	return array(poly)*array(rho).repeat(m,0).repeat(m,1)

def amb(z, h, rho):
	"""Calculate the A-B terms in the Levenberg-Marquardt H, between amplitudes z and z+h.
	rho gives the inner products between states <z+h| and |z>"""
	n, m, f, A = extract(matrix(z))
	n, m, df, dA = extract(matrix(h))
	expt = array(df+dA*(A+dA).H)
	w = array(dz)
	w[:,0] = 0
	w = w.reshape(-1,1)
	v = array(z)
	v[:,0] = 1
	v = w.reshape(-1,1)
	return array(rho).repeat(m,0)*(w*exp(expt).repeat(m,0) + v*expm1(expt).repeat(m,0))
	
def jham(z, h, rho):
	"""Calculate the Hamiltonian bracket in the Levenberg-Marquardt H, between amplitudes z and z+h.
	rho gives the inner products between states <z+h| and |z>"""

def jfock(z, cs):
	"Calculate the bracket between the Jacobian and a ket expanded over Fock states.  The Fock amplitudes should be a 1D ndarray."
	n, m, f, a = extract(z)
	Q = empty((2*n,1))
	# the Python monkeys got the columns of Vandermonde matrices backwards, so arrays of coefficients go backwards too.
	evens = cs/sqrt(factorial(xrange(cs.size)))
	evens = evens[::-1]
	Q[0:2*n:2,:] = vander(a.conjugate()[:,0], evens.size)*matrix(evens).T
	odds = cs[1:]*sqrt(xrange(1,cs.size)/factorial(xrange(cs.size-1)))
	odds = odds[::-1]
	Q[1:2*n:2,:] = vander(a.conjugate()[:,0], odds.size)*matrix(odds).T
	Q = array(Q)*exp(f.conjugate()).repeat(2,axis=0)
	return Q.reshape((Q.size,))
