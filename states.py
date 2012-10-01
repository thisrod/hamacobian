# refactored matrices, bras and kets
# implement bottom up

from numpy import ndarray, array, newaxis, empty, zeros, ones, arange, eye, diagflat, vander, transpose, concatenate, squeeze, log, sqrt, exp, conjugate, sum, cumsum, dot, maximum, vstack, hstack
from scipy.misc import factorial
import numbers
from copy import copy, deepcopy

##################################################
#
#	operator dispatch
#
##################################################

class QO(object):

	sums = {}
	products = {}
			
	def __rmul__(self, z):
		if isinstance(z, numbers.Complex):
			return deepcopy(self).scale(z)
		else:
			return NotImplemented
		
		
def operator(optable):
	def op(self, other):
		candidates = [optable[x,y] for x,y in optable.iterkeys()
			if isinstance(self, x) and isinstance(other, y)]
		assert len(candidates) < 2
		if candidates:
			return candidates[0](self, other)
		else:
			return NotImplemented
	return op
	
QO.__add__ = operator(QO.sums)
QO.__mul__ = operator(QO.products)

QO.products[QO, numbers.Complex] = \
	lambda Q, z: deepcopy(Q).scale(z.conjugate())

		

##################################################
#
#	quantum states
#
##################################################

class State(QO):

	def __rmul__(self, other):
		if isinstance(other, State):
			# danger of infinite recursion
			return operator(QO.products)(self, other).conj().T
		else:
			return QO.__rmul__(self, other)

def wid(s):
	"The width of a state is the number of parameters defining it."
	return s.__wid__()

class FockExpansion(State):
	"the common case is n==0 or 1"
	
	def __init__(self, cs):
		"cs is a 1darray of coefficients."
		self.prms = array(cs, dtype=complex)
		
	def __wid__(self):
		return self.prms.size
		
	def scale(self, z):
		self.prms *= z
		return self
				
	def lowered(self):
		return FockExpansion(self.prms[1:]*sqrt(xrange(1,wid(self))))
		
	def raised(self):
		rprms = sqrt(arange(wid(self)+1, dtype=complex))
		rprms[1:] *= self.prms
		return FockExpansion(rprms)

def NNadd(self, other):
	# silently discarding imaginary parts: yet another interesting designe choice from Numpy
	prms = zeros(max(wid(self), wid(other)), dtype=complex)
	for x in self, other: prms[:wid(x)] += x.prms
	return FockExpansion(prms)
		
def NNmul(bra, ket):
	a = bra.prms.conj()
	b = eye(wid(bra), wid(ket))
	c = ket.prms
	return dot(dot(a,b),c)
	
QO.sums[FockExpansion,FockExpansion] = NNadd
QO.products[FockExpansion,FockExpansion] = NNmul

	
class DisplacedState(State):
	"a state with a displacement operator applied to it, log scaled.  The expansion of the state over number states must be finite."
	
	def __init__(self, state, f, a):
		self.base = state
		self.f = complex(f)
		self.a = complex(a)
		
	def __wid__(self):
		return 2
		
	def scale(self, z):
		self.f += log(z)
		return self
		
	def raised(self):
		"commute through the displacement operator"
		return DisplacedState(self.base.raised() + self.base*self.a.conjugage(), self.f, self.a)
		
	def lowered(self):
		"commute through the displacement operator"
		return DisplacedState(self.base.lowered() + self.base*self.a, self.f, self.a)
		
def explower(state, a):
	"exp(a*lower) state"
	n, Q, rslt = 2, a * state.lowered(), state
	while abs(Q*Q) > 0:
		n, Q, rslt = n+1, (a/n)*Q.lowered(), rslt+Q
	return rslt
	
def DAmul(Dbra, ket):
	return exp(Dbra.f.conjugate() - 0.5*abs(Dbra.a)**2) * \
		explower(Dbra.base, -Dbra.a.conjugate()) * \
		explower(ket, Dbra.a.conjugate())
	
def DDmul(Dbra, Dket):
	a, b = Dbra.a, Dket.a
	return exp(Dbra.f.conjugate() + Dket.f - \
		0.5*abs(a)**2 - 0.5*abs(b)**2 + a.conjugate()*b) * \
		explower(Dbra.base, (b-a).conjugate()) * \
		explower(Dket.base, (a-b).conjugate())
		
QO.products[DisplacedState,FockExpansion] = DAmul
QO.products[DisplacedState,DisplacedState] = DDmul


##################################################
#
#	things to put in ndarrays
#
##################################################

class BraSum(QO):
	pass
	
class KetSum(QO):
	pass
