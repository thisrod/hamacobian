from brackets import combinations
from numpy import array, empty, zeros_like, eye, diagflat, transpose, concatenate, squeeze, log, sqrt, conjugate, sum, cumsum, dot
import numbers
from copy import copy, deepcopy


##################################################
#
#	arithmetic dispatch
#
##################################################

class QO(object):
	"I dispatch arithmetic operations.  subclasses should implement scaled, for scalar multiplication."
	
	 # these dispatch tables are set up at the end of this file, when the classes they refer to are defined.
	sums = {}
	products = {}
	
	def _for(self, other, ops):
		return [ops[x,y] for x,y in ops.iterkeys() if isinstance(self, x) and isinstance(other, y)]

	def __add__(self, other):
		ops = self._for(other, QO.sums)
		assert len(ops) < 2
		if ops:
			return ops[0](self, other)
		else:
			return NotImplemented

	def __mul__(self, other):
		ops = self._for(other, QO.products)
		assert len(ops) < 2
		if ops:
			return ops[0](self, other)
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		assert isinstance(other, numbers.Complex)
		return self*other 
			
	def __div__(self, z):
		if isinstance(z, numbers.Complex):
			return deepcopy(self).scale(1./z)
		else:
			return NotImplemented
			
	# These work through references if possible	
	def __imul__(self, z):
		assert isinstance(z, numbers.Complex)
		return self.scale(z)
			
	def __idiv__(self, z):
		assert isinstance(z, numbers.Complex)
		return self.scale(1./z)
		
	
##################################################
#
#	states base class
#
##################################################

def wid(s):
	"The width of a collection of states is the number of parameters defining each state."
	return s.__wid__()

class States(QO):
	"a sequence of quantum states.  multiplication gives an array of inner products"
	
	# subclasses set params to a list of parameters for each state
	# by convention, an orthonormal expansion calls its coefficients "c"
	params = []
	vals = None
	
	def __init__(self, *args):
		# there is an arg for each var of each state.  the actual parameters are stored flat as the rows of the 2D array vals, and instances have attributes that are views into it
		args = (array(x, ndmin=1) for x in args)
		rows = zip(*[args]*len(self.params))
		self._divs = cumsum([0] + [x.size for x in rows[0]])
		self._setvals(array([concatenate(c) for c in rows], dtype = complex))
					
	# subclasses should implement slicing, but not indexing.  that's done in KetRow or BraCol, which know if their elements are bras or kets.
	
	def __len__(self):
		return self.vals.shape[0]
		
	def __wid__(self):
		return self.vals.shape[1]
	
	def _setvals(self, vals):
		"replace vals with a similar set, and make parameter attributes consistent"
		assert self.vals is None or vals.shape[1] == wid(self)
		self.vals = vals
		for i in xrange(len(self._divs)-1):
			setattr(self, self.params[i], self.vals[:,self._divs[i]:self._divs[i+1]])
		return self
		
	def __deepcopy__(self, d):
		return copy(self)._setvals(self.vals.copy())
		
	def bras(self):
		return BraCol(self)
		
	def kets(self):
		return KetRow(self)
		
	def sum(self):
		return KetSum(self)
			
	# state multiplication exploits <a|b> = <b|a>*
	def __rmul__(self, other):
		if isinstance(other, States):
			ops = self._for(other, QO.products)
			assert len(ops) < 2
			if ops:
				return ops[0](self, other).conj().T
		else:
			return QO.__rmul__(self, other)
	
	
	
##################################################
#
#	row, column and sum classes
#
##################################################

class BraCol(QO):
	
	def __init__(self, states):
		self.elements = states
		
	def sum(self):
		return BraSum(self.elements)
		
	def __len__(self):
		return len(self.elements)
		
	def __getitem__(self, i):
		return BraSum(self.elements[i:i+1])
		
	def scale(self, z):
		# Dirac notation obscures this conjugate except when z is an eigenvalue. 
		self.elements.scale(z.conjugate())
		return self

	def H(self):
		return KetRow(self.elements)
	

class KetRow(QO):
	
	def __init__(self, states):
		self.elements = states
		
	def sum(self):
		return KetSum(self.elements)
		
	def __len__(self):
		return len(self.elements)
		
	def __getitem__(self, i):
		return KetSum(self.elements[i:i+1])
		
	def scale(self, z):
		self.elements.scale(z)
		return self

	def H(self):
		return BraCol(self.elements)
		
class BraSum(QO):
	
	# if a sum has one term, the params of components become attributes of the sum, with axis 1 elided.
	
	def __init__(self, states):
		self.terms = states
		if len(states) == 1:
			for p in states.params:
				setattr(self, p, getattr(states, p).flatten())
				
	def D(self):
		return BraCol(self.terms.D())
		
	def scale(self, z):
		self.terms.scale(z.conjugate())
		return self
		
	def H(self):
		return KetSum(self.terms)
	
class KetSum(QO):

	def __init__(self, states):
		self.terms = states
		if len(states) == 1:
			for p in states.params:
				setattr(self, p, getattr(states, p).flatten())
				
	def D(self):
		return KetRow(self.terms.D())
		
	def scale(self, z):
		self.terms.scale(z)
		return self
		
	def H(self):
		return BraSum(self.terms)
	
	
##################################################
#
#	specific types of quantum state
#
##################################################

class FockStates(States):
	"c[n] is the coefficient of |n>"
	
	params = ["c"]
	
	def scale(self, z):
		self.c *= z
		return self
	
	def __getitem__(self, ns):
		raise NotImplementedError		

class CoherentStates(States):
	# f is a log weight, and a a vector ampltiude.  ideally, log weights would be stored with a floating point real part, and a fixed point imaginary part
	params = ["f", "a"]

	def __getitem__(self, ns):
		assert isinstance(ns, slice)
		# try slice(1000, 1001).indices(10)
		m, n = ns.start, ns.stop
		if min(m,n) < 0 or max(m,n) > len(self):
			raise IndexError
		return copy(self)._setvals(self.vals[ns, :])
		
	def scale(self, z):
		self.f += log(z)
		return self
		
	def D(self):
		return DCoherentStates(self)
				

class DCoherentStates(States):
	"the sequence of partial derivative of CoherentStates"
	
	params = CoherentStates.params
	
	def __init__(self, cstates):
		self._divs = cstates._divs
		# scaling a derivative doesn't scale the point at which it was taken
		self._setvals(cstates.vals.copy())
		
	scale = CoherentStates.scale
		
	def __len__(self):
		return self.vals.size
		
	def __wid__(self):
		# No adjustable parameters
		return 0
		
		
##################################################
#
#	Dispatch table
#
##################################################

QO.products[QO, numbers.Complex] = \
	lambda Q, z: deepcopy(Q).scale(z)

QO.products[BraCol, KetRow] = \
	lambda bras, kets: bras.elements * kets.elements
	
QO.products[BraSum, KetRow] = \
	lambda bra, kets: sum(bra.terms * kets.elements, axis=0)
	
QO.products[BraCol, KetSum] = \
	lambda bras, ket: sum(bras.elements * ket.terms, axis=1)
	
QO.products[BraSum, KetSum] = \
	lambda bra, ket: sum(bra.terms * ket.terms)
	
	
# The values of a given parameter attribute for all the states forms a column.  So we usually need to transpose those from the ket, and conjugate those from the bra

def bracket(f):
	return lambda bras, kets: \
		f(Params(conjugate, bras), Params(transpose, kets))
		
class Params:
	# dear reader: I can't do this with new style classes.  can you?
	def __init__(self, F, bks):
		assert isinstance(bks, States)
		self._prototype = bks
		self.F = F
			
	def __getattr__(self, name):
		v = getattr(self._prototype, name)
		return self.F(v) if name == "vals" or name in self._prototype.params else v
		
@bracket
def mul_orth(bras, kets):
	"product of expansions over the same orthogonal basis"
	a = diagflat(bras.c)
	b = eye(len(bras), len(kets))
	c = diagflat(kets.c)
	return dot(dot(a,b),c)
	
QO.products[FockStates,FockStates] = mul_orth
	
for Ts, op in combinations(FockStates, CoherentStates, DCoherentStates):
	QO.products[Ts] = bracket(op)
