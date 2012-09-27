from numpy import array, empty, vander, exp, sqrt, dot, newaxis
from scipy.misc import factorial

# the slice [::-1] means reverse order; Numpy puts the columns of Vandermonde matrices backwards, so it appears often.

# operators are decorated with @bracket as they enter the dispatch table in states.py.

# Hungarian prefixes:
# N = FockStates, C = CoherentStates, D = DCoherentStates

def wid(x):
	return x.__wid__()
	
def NCmul(Nbras, Ckets):
	if wid(Ckets) > 2:
		raise NotImplementedError, "Fock states are single mode only"
	n = len(Nbras)
	return vander(Ckets.a.flatten(), n)[:,::-1].T  / \
		sqrt(factorial(xrange(n)))[:,newaxis] * \
		Nbras.c * exp(Ckets.f)
		
def CCmul(bras, kets):
	return exp(bras.f + kets.f + dot(bras.a, kets.a))
			
def NDmul(Nbras, Dkets):
	if Dkets.a.shape[1] > 1:
		raise NotImplementedError, "Fock states are single mode only"
	ns = array(xrange(1,len(other)), ndmin=2)
	result = empty((len(self), len(other)), dtype=complex)
	result[0::2,:] = self.components*other
	r = result[1::2,:]
	r[:,0] = 0
	r[:,:0:-1] = vander(hc(self.components.a).flatten(), len(other)-1)
	r[:,1::] *= other.cs[:,1::]*sqrt(ns/factorial(ns-1))
	r *= hc(exp(self.components.f))
	return result
	
def mulDL(self, other):		# FIXME
	if wid(self.components) > 2 or self.components is not other:
		raise NotImplementedError, "Not yet needed"
	result = empty((len(self),len(other)), dtype=complex)
	Z = self.components
	n = len(Z)
	m = wid(Z)
	rho = (Z*Z)
	w = Z.z.copy()
	w[0,:] = 1
	w = w.reshape(1,-1)
	poly = dot(hc(w), w)
	result[0::2] = sum(rho, 0)
	result[1::2] = dot(rho, other.a)
	return result
	
def mulDD(self, other):		# FIXME
	# V matrix, <other|self>
	if other.components is not self.components:
		raise NotImplementedError, "Not yet needed"
	Z = self.components
	n = len(Z)
	m = wid(Z)
	rho = (Z*Z).repeat(m,0).repeat(m,1)
	w = Z.z.copy()
	w[:,0] = 1
	w = w.reshape(-1,1)
	poly = dot(w, hc(w))
	for i in xrange(1,m):
		poly[i:n:n*m,i:n:n*m] += 1
	return poly*rho
	
def combinations(n, c, d):
	"yield a sufficient set of (ltype, rtype), op tuples"
	
	yield (n, c), NCmul
	yield (c, c), CCmul
	yield (n, d), NDmul
