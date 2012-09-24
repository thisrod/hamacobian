# Linear combinations of coherent states, and their moments.

from numpy import ndarray, array, empty, diag, dot, vander, exp, sqrt, sum
from scipy.misc import factorial
import numbers
	

class KetRow(object):
	
	def similar(self, other):
		return type(self) is type(other) and \
			len(self) == len(other) and \
			wid(self) == wid(other)
			
	def norm(self):
		return sqrt(sum(self*self).real)

def wid(s):
	return s.__wid__()

# columns of bras need conjugate parameters.  thus col must make a copy; for consistency, row does too.

def col(x):
	return hc(row(x))

def hc(A):
	"our very own Hermitian conjugate function.  hooray for Numpy!"
	return A.conjugate().T

def row(x):
	return array(x, copy=True, ndmin=2)


class LccState(KetRow):
	"A linear combination of coherent states."

	def __init__(self, *args):
		"LccState(f1, a1, ..., fn, an) where f is a logarithmic weight, and a the corresponding coherent amplitude(s)."
		# Use copy construction if efficiency matters
		if isinstance(args[1], numbers.Complex):
			ain = [[x] for x in args[1::2]]
		else:
			ain = args[1::2]
		n = len(args)/2
		m = len(ain[0])+1
		self.z = empty((n, m), dtype=complex)
		self.f = self.z[:,0]
		self.a = self.z[:,1:]
		self.setf(args[0::2])
		self.seta(ain)
		
	def __len__(self):
		return self.z.shape[0]
		
	def __wid__(self):
		return self.z.shape[1]
		
	def setf(self, f):
		self.z[:,0] = f
		
	def seta(self, a):
		self.z[:,1:] = a
		
	def D(self):
		result = DLccState()
		result.be(self)
		return result
		
	def __add__(self, z):
		result = LccState(*z)
		result.z += self.z
		return result
		
	def __mul__(self, other):
		return other.mulL(self)
		
	def mulL(self, other):		# FIXME
		# other * self
		assert self.similar(other)
		return exp(row(other.f) + col(self.f) + dot(self.a, hc(other.a)))
		
	def mulD(self, other):
		# <D other|self>
		if wid(self) > 2 or self is not other:
			raise NotImplementedError, "Not yet needed"
		Q = empty((2*len(self),))
		rho = other*self
		Q[0::2] = sum(rho, 1)
		Q[1::2] = dot(rho, self.a)
		return Q

class DLccState(KetRow):
	"the total derivative of an LccState wrt z.  forms products with states and number state vectors as 2D arrays."

	def be(self, state):
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
	
	def __init__(self, *cs):
		"cs are the coefficients, starting with |0>"
		self.cs = array(cs, dtype=complex)
		assert self.cs.ndim == 1
		
	def __len__(self):
		return self.cs.shape[0]
		
	def __wid__(self):
		return 1
		
	def __mul__(self, other):
		return other.mulN(self)
		
	def D(self):
		return self

	def mulN(self, other):
		x = other.cs.conjugate()
		y = self.cs.copy()
		if y.size > x.size:
			x, y = y, x
		x[:y.size] *= y
		result = diag(x)
		return result[:len(self), :len(other)]

	def mulL(self, other):
		if wid(other) > 2:
			raise NotImplementedError, "Not yet needed"
		n = len(self)
		result = vander(col(other.a).flatten(), n)[:,::-1]
		result *= row(self.cs)/sqrt(factorial(xrange(n)))
		result *= col(exp(other.f))
		return result
				
	def mulD(self, other):
		# the Python monkeys got the columns of Vandermonde matrices backwards
		if wid(other) > 2:
			raise NotImplementedError, "Not yet needed"
		ns = col(xrange(1,len(self)))
		result = empty((len(self), 2*len(other)), dtype=complex)
		result[:, 0::2] = other*self
		r = result[:,1::2]
		r[0,:] = 0
		r[:0:-1,:] = vander(other.a.conjugate()[:,0], len(self)-1).T
		r[1::,:] *= col(self.cs[1::])*sqrt(ns/factorial(ns-1))
		r *= row(exp(other.f))
		return result

# Test data

glauber = LccState(0, 1.8)
basis = NState(*[1]*15)
# fock = NState(*glauber*basis)  NYI
fock = NState(*(glauber*basis).flatten().conj())