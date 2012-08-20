#
# Compute the inner product form, the squared Jacobian, and the Hamiltonian moments in the obvious way, using interpreted for loops.
#
# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.

from numpy import array, matrix, exp

	
def extract(z):
	# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.
	return z.shape + (z[:,0:1], z[:,1:])
	
def rho(z):
	n, m, f, A = extract(matrix(z))
	logrho = f.H + f + A*A.H
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
