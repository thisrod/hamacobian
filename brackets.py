from numpy import array, empty, diag, vander, exp, sqrt, dot
from scipy.misc import factorial

# the Python monkeys got the columns of Vandermonde matrices backwards

def hc(A):
	"our very own Hermitian conjugate function.  hooray for Numpy!"
	return A.conjugate().T
	
def wid(x):
	return x.__wid__()
		
def mulLL(self, other):
	if not self.similar(other):
		raise NotImplementedError, "Not yet needed"
	return exp(hc(self.f) + other.f + dot(hc(self.a), other.a))
	
def mulDL(self, other):		# FIXME
	if wid(self.components) > 2 or self.components is not other:
		raise NotImplementedError, "Not yet needed"
	Z = self.components
	n = len(Z)
	m = wid(Z)
	rho = (Z*Z).repeat(m,0)
	w = Z.z.copy()
	w[:,0] = 1
	w = w.reshape(-1,1)
	poly = dot(w, hc(w))
	result = empty((len(self.components),len(other)), dtype=complex)
	Q[0::2] = sum(rho, 1)
	Q[1::2] = dot(rho, other.a)
	return Q
	
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

def mulNN(self, other):
	x = hc(self.cs).flatten()
	y = other.cs.flatten()
	if y.size > x.size:
		x, y = y, x
	x[:y.size] *= y
	return diag(x)[:len(other), :len(self)]
	
def mulLN(self, other):
	if wid(self) > 2:
		raise NotImplementedError, "Not yet needed"
	n = len(other)
	result = vander(hc(self.a).flatten(), n)[:,::-1]
	result *= other.cs/sqrt(factorial(xrange(n)))
	result *= hc(exp(self.f))
	return result
			
def mulDN(self, other):
	if wid(self.components) > 2:
		raise NotImplementedError, "Not yet needed"
	ns = array(xrange(1,len(other)), ndmin=2)
	result = empty((len(self), len(other)), dtype=complex)
	result[0::2,:] = self.components*other
	r = result[1::2,:]
	r[:,0] = 0
	r[:,:0:-1] = vander(hc(self.components.a).flatten(), len(other)-1)
	r[:,1::] *= other.cs[:,1::]*sqrt(ns/factorial(ns-1))
	r *= hc(exp(self.components.f))
	return result
	
def combinations(N, L, D):
	"yield a sufficient set of (ltype, rtype), op tuples"
	yield (L, L), mulLL
	yield (D, D), mulDD
	yield (D, L), mulDL
	yield (N, N), mulNN
	yield (L, N), mulLN
	yield (D, N), mulDN