########################################
#
#	Product functions
#
#	The upper triangle of the DLN matrix is implemented explicitly.
#	The remainder is done by lum.
#
########################################

from numpy import array, empty, diag, vander, exp, sqrt, dot
from scipy.misc import factorial

def hc(A):
	"our very own Hermitian conjugate function.  hooray for Numpy!"
	return A.conjugate().T
	
def wid(x):
	return x.__wid__()
	
# double dispatch: implement other*self
	
def lum(self, other):
	"Do it the other way round, and take the conjugate"
	return hc(self*other)
		
def mulLL(self, other):
	if not self.similar(other):
		raise NotImplementedError, "Not yet needed"
	return exp(hc(other.f) + self.f + dot(hc(other.a), self.a))
	
def mulDL(self, other):		# FIXME
	# <D other|self>
	if wid(self) > 2 or self is not other:
		raise NotImplementedError, "Not yet needed"
	Q = empty((2*len(self),))
	rho = other*self
	Q[0::2] = sum(rho, 1)
	Q[1::2] = dot(rho, self.a)
	return Q
	
def mulDD(self, other):
	# V matrix, <other|self>
	if other is not self.state:
		raise NotImplementedError, "Not yet needed"
	Z = self.state
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
	x = hc(other.cs).flatten()
	y = self.cs.flatten()
	if y.size > x.size:
		x, y = y, x
	x[:y.size] *= y
	return diag(x)[:len(self), :len(other)]

	# the Python monkeys got the columns of Vandermonde matrices backwards
	
def mulLN(self, other):
	if wid(other) > 2:
		raise NotImplementedError, "Not yet needed"
	n = len(self)
	result = vander(hc(other.a).flatten(), n)[:,::-1]
	result *= self.cs/sqrt(factorial(xrange(n)))
	result *= hc(exp(other.f))
	return result
			
def mulDN(self, other):
	if wid(other) > 2:
		raise NotImplementedError, "Not yet needed"
	ns = array(xrange(1,len(self)), ndmin=2)
	result = empty((2*len(other), len(self)), dtype=complex)
	result[0::2,:] = other*self
	r = result[1::2,:]
	r[:,0] = 0
	r[:,:0:-1] = vander(hc(other.a).flatten(), len(self)-1)
	r[:,1::] *= self.cs[:,1::]*sqrt(ns/factorial(ns-1))
	r *= hc(exp(other.f))
	return result