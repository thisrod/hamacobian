#
# Compute the inner product form, the squared Jacobian, and the Hamiltonian moments in the obvious way, using interpreted for loops.
#
# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.

from numpy import array, matrix, empty, exp, sum

	
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

def jlft(z, rho):
	"""Calculate the product of the Jacobian bra matrix with the state.  Limited to one mode."""
	n, m, f, a = extract(matrix(z))
	Q = empty((2*n,1))
	Q[0:2*n:2,:] = sum(rho, 1)
	Q[1:2*n:2,:] = rho*a
	return Q
