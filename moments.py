#
# Compute the inner product form, the squared Jacobian, and the Hamiltonian moments in the obvious way, using interpreted for loops.
#
# Convention: f stores a column vector of phis, A stores a matrix whose rows are alpha vectors.
# convention: moments have type matrix, not array.

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
