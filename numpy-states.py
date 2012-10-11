# refactored matrices, bras and kets
# implement bottom up

from numpy import ndarray, array, newaxis, empty, zeros, ones, arange, eye, diagflat, vander, transpose, concatenate, squeeze, reshape, log, sqrt, exp, conjugate, sum, cumsum, dot, maximum, vstack, hstack
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
			return self.scaled(z)
		else:
			return NotImplemented
			
	def __sub__(self, other):
		return self+(-other)
		
	def __neg__(self):
		return self.scale(-1)
			
	def scaled(self,z):
		return deepcopy(self).scale(z)
		
		
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
	lambda Q, z: Q.scaled(z.conjugate())

		

##################################################
#
#	quantum states
#
#	N.B. states reject scalar multiplication, because it's too easy to say
#	z * bra * ket when you mean bra * (z * ket).
#	Multiply complex numbers, or use .scale() if you must.
#
##################################################

class State(QO):

	def __mul__(self, other):
		assert not isinstance(other, numbers.Complex)
		return demote(QO.__mul__(self, other))

	def __rmul__(self, other):
		assert not isinstance(other, numbers.Complex)
		if isinstance(other, State):
			# danger of infinite recursion
			return demote(conjugate(operator(QO.products)(self, other)))
		else:
			return demote(QO.__rmul__(self, other))
				
def demote(zs):
	return array(zs, dtype=complex)


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
		
	def isvac(self):
		return all(self.prms[1:] == 0)

def NNadd(self, other):
	# silently discarding imaginary parts: yet another interesting design choice from Numpy
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
	
	def __init__(self, state, a=0):
		self.base = state
		self.a = complex(a)
		
	def __wid__(self):
		return 2
		
	def scale(self, z):
		self.base.scale(z)
		return self
		
	def raised(self):
		"commute through the displacement operator"
		return DisplacedState(self.base.raised() + self.base.scaled(self.a.conjugate()), self.a)
		
	def lowered(self):
		"commute through the displacement operator"
		return DisplacedState(self.base.lowered() + self.base.scaled(self.a), self.a)
		
def DDadd(self, other):
	if self.a == other.a:
		return DisplacedState(self.base+other.base, self.a)
	else:
		return Sum([self, other])
		
def DNadd(self, other):
	return self + DisplacedState(other)
		
def explower(state, a):
	"exp(a*lower) state"
	n, Q, rslt = 2, state.lowered().scale(a), state
	while abs(Q*Q) > 0:
		n, Q, rslt = n+1, Q.lowered().scale(a/n), rslt+Q
	return rslt
	
def DAmul(Dbra, ket):
	return exp(-0.5*abs(Dbra.a)**2) * \
		(explower(Dbra.base, -Dbra.a.conjugate()) * \
		explower(ket, Dbra.a.conjugate()))
	
def DDmul(Dbra, Dket):
	a, b = Dbra.a, Dket.a
	return exp(-0.5*abs(a)**2 - 0.5*abs(b)**2 + a.conjugate()*b) * \
		(explower(Dbra.base, (b-a).conjugate()) * \
		explower(Dket.base, (a-b).conjugate()))
		
QO.sums[DisplacedState,FockExpansion] = DNadd
QO.sums[DisplacedState,DisplacedState] = DDadd
QO.products[DisplacedState,FockExpansion] = DAmul
QO.products[DisplacedState,DisplacedState] = DDmul


class Sum(State):
	"a sum of DisplacedStates, with different amplitudes"
	
	def __init__(self, terms):
		assert all(isinstance(t, DisplacedState) for t in terms)
		self.terms = dict((t.a, t) for t in terms)
		
	def scale(self, z):
		for q in self.terms.values():
			q.scale(z)
		return self
		
	def asstvac(self):
		assert all(q.base.isvac() for q in self.terms.values())
			
	def prms(self):
		self.asstvac()
		return array([(log(q.base.prms[0]), q.a) for q in self.terms.values()]).flatten()
				
	def smrp(self, zs):
		states = [ DisplacedState(FockExpansion([exp(f)]), a)
			for f, a in reshape(array(zs), (-1,2)) ]
		return Sum(states)
		
	def D(self):
		self.asstvac()
		return array(zip(self, self.raised())).flatten()
		
	def raised(self):
		return Sum([q.raised() for q in self.terms.values()])
		
	def lowered(self):
		return Sum([q.lowered() for q in self.terms.values()])
	
			
def SDadd(Sself, Dother):
	qs = copy(Sself.terms)
	if Dother.a in qs:
		qs[Dother.a] = qs[Dother.a] + Dother
	else:
		qs[Dother.a] = Dother
	return Sum(qs.values())
	
def SNadd(self, other):
	return self + DisplacedState(other)
	
def SSadd(Sself, Sother):
	rslt = Sself
	for t in Sother.terms.values():
		rslt += t
	return rslt
	
def SAmul(Sbra, ket):
	return sum(bra * ket for bra in Sbra.terms.values())
	
QO.products[Sum,State] = SAmul
QO.sums[Sum, DisplacedState] = SDadd
QO.sums[Sum, FockExpansion] = SNadd
QO.sums[Sum, Sum] = SSadd


##################################################
#
#	bras and kets check types, and track conjugation
#
##################################################

class Braket(QO):
	def __init__(self, s):
		self.s = s
		
	def prms(self):
		return self.s.prms()
		
	def smrp(self, zs):
		return type(self)(self.s.smrp(zs))

class Bra(Braket):
	def scale(self, z):
		# this conjugate makes Dirac multiplication associative
		self.s.scale(conjugate(z))
		return self
		
	def D(self):
		return array([Bra(q) for q in self.s.D()])[:,newaxis]
	
class Ket(Braket):
	def scale(self, z):
		self.s.scale(z)
		return self
		
	def D(self):
		return array([Ket(q) for q in self.s.D()])[newaxis,:]


def BKmul(bra, ket):
	return bra.s * ket.s
	
def BKadd(self, other):
	return type(self)(self.s + other.s)
	
QO.sums[Bra,Bra] = BKadd
QO.sums[Ket,Ket] = BKadd
QO.products[Bra,Ket] = BKmul
