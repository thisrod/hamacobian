# Linear combinations of coherent states, and their moments.

from numpy import array, empty, dot, exp

def col(x):
	"Cast a 1D array to a column"
	return array(x, copy=True, ndmin=2).T

def hc(A):
	"our very own Hermitian conjugate function.  hooray for Numpy!"
	return A.conjugate().T

def row(x):
	"Cast a 1D array to a row, taking the complex conjugate"
	return hc(col(x))
	
def lccs(*args):
	"lccs(f1, a1, ..., fn, an) where f is a logarithmic weight, and a the corresponding coherent amplitude(s), returns the state object."
	if isinstance(args[1], float):
		ain = [[z] for z in args[1::2]]
	else:
		ain = args[1::2]
	n = len(args)/2
	m = len(ain[0])+1
	z = empty((n, m), dtype=complex)
	instance = LccState(z)
	instance.setf(args[0::2])
	instance.seta(ain)
	return instance
	


class LccState(object):
	"A linear combination of coherent states."

	def __init__(self, z):
		"z must be an ndarray"
		self.n, self.m = z.shape
		self.z = z
		self.f = z[:,0]
		self.a = z[:,1:]
		
	def setf(self, f):
		self.z[:,0] = f
		
	def seta(self, a):
		self.z[:,1:] = a
		
	def similar(self, w):
		"do I have the same number of modes and components as w?"
		return self.n == w.n and self.m == w.m
		
	def D():
		"my derivative"
		return DLccState(self)
		
	def rho(self, w=None):
		"""the array of inner products between the components of self and w."""
		if w is None: w = self
		assert self.similar(w)
		return exp(row(self.f) + col(w.f) + dot(w.a, hc(self.a)))
		

class DLccState(object):
	"the total derivative of an LccState wrt z.  forms products with states and number state vectors as 2D arrays."

	def __init__(self, state):
		self.Q = state