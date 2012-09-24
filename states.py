# Linear combinations of coherent states, and their moments.

from numpy import array, empty, zeros, dot, exp
	
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
	pass

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
		"z must be an ndarray"
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
		
	def similar(self, w):
		"do I have the same number of modes and components as w?"
		return self.n == w.n and self.m == w.m
		
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
		if self.m > 2 or self is not other:
			raise NotImplementedError, "Not yet needed"
		Q = empty((2*self.n,))
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
		return 0
		
	def __mul__(self, other):
		return other.mulD(self.state)
		
	def mulD(self, other):
		# V matrix, <other|self>
		if other is not self.state:
			raise NotImplementedError, "Not yet needed"
		Q = self.state
		n = Q.n
		m = Q.m
		rho = (Q*Q).repeat(m,0).repeat(m,1)
		w = Q.z.copy()
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
		self.cs = cs
		
	def __len__(self):
		return cs.shape[0]
		
	def __wid__(self):
		return 1
		
	def __mul__(self, other):
		return other.mulN(self)

	def mulN(self, other):
		result = zeros((len(self.cs), len(other.cs)))

	def mulL(self, other):
		raise NotImplementedError, "Not yet"
				
	def mulD(self, other):
		raise NotImplementedError, "Not yet"

