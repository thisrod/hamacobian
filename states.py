# Linear combinations of coherent states, and their moments.

from numpy import ndarray, array, empty, zeros, diag, dot, exp, sum
from math import sqrt
	
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

class KetRow(object):
	
	def similar(self, other):
		return type(self) is type(other) and \
			len(self) == len(other) and \
			wid(self) == wid(other)
			
	def norm(self):
		return sqrt(sum(self*self).real)

def wid(s):
	return s.__wid__()

def col(x):
	"Cast a 1D array to a column"
	return array(x, copy=True, ndmin=2).T

def hc(A):
	"our very own Hermitian conjugate function.  hooray for Numpy!"
	return A.conjugate().T

def row(x):
	"Cast a 1D array to a row, taking the complex conjugate"
	return hc(col(x))


class LccState(KetRow):
	"A linear combination of coherent states."

	def __init__(self, z):
		assert type(z) is ndarray and len(z.shape) == 2
		self.z = z
		self.f = z[:,0]
		self.a = z[:,1:]
		
	def __len__(self):
		return self.z.shape[0]
		
	def __wid__(self):
		return self.z.shape[1]
		
	def setf(self, f):
		self.z[:,0] = f
		
	def seta(self, a):
		self.z[:,1:] = a
		
	def D(self):
		"my derivative"
		return DLccState(self)
		
	def __mul__(self, other):
		return other.mulL(self)
		
	def mulL(self, other):
		# other * self
		assert self.similar(other)
		return exp(row(other.f) + col(self.f) + dot(self.a, hc(other.a)))
		
	def mulD(self, other):
		# <D other|self>
		if wid(self) > 2 or self is not other:
			raise NotImplementedError, "Not yet needed"
		Q = empty((2*len(self),))
		rho = other*self
		Q[0:2*n:2] = sum(rho, 1)
		Q[1:2*n:2] = dot(rho, self.a)
		return Q

class DLccState(KetRow):
	"the total derivative of an LccState wrt z.  forms products with states and number state vectors as 2D arrays."

	def __init__(self, state):
		self.state = state
		
	def __len__(self):
		return len(self.state)*wid(self.state)
		
	def __wid__(self):
		# No adjustable parameters
		return 0
		
	def __mul__(self, other):
		return other.mulD(self.state)
		
	def mulD(self, other):
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


class NState(KetRow):
	"a state expanded over Fock states."
	
	def __init__(self, cs):
		"cs are the coefficients, starting with |0>"
		self.cs = array(cs, dtype=complex)
		assert self.cs.ndim == 1
		
	def __len__(self):
		return self.cs.shape[0]
		
	def __wid__(self):
		return 1
		
	def __mul__(self, other):
		return other.mulN(self)

	def mulN(self, other):
		x = other.cs.conjugate()
		y = self.cs.copy()
		if y.size > x.size:
			x, y = y, x
		x[:y.size] *= y
		result = diag(x)
		return result[:len(self), :len(other)]

	def mulL(self, other):
		raise NotImplementedError, "Not yet"
				
	def mulD(self, other):
		raise NotImplementedError, "Not yet"

