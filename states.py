"""
Dirac algebra

This module implements bras, kets, linear operators and matrices of
those things.
"""

from cmath import exp, log
from math import sqrt
import numbers

import numpy


# Matrices

def row(x):
	"""Return a 1*len(x) Matrix sharing elements with x."""
	x = list(x)
	return Matrix(1, len(x), x)

def col(x):
	"""Return a len(x)*1 Matrix sharing elements with x."""
	x = list(x)
	return Matrix(len(x), 1, x)

def dot(xs, ys):
	"""Return the complex inner product of two sequences.

	The elements of the first sequence are conjugated. If either
	sequence is empty, return 0j.
	"""
	if min(len(xs), len(ys)) is 0:
		return 0j
	else:
		return reduce(
			lambda a,b: a+b,
			(x.conjugate()*y for (x, y) in zip(xs, ys)))
	
def _matrix_scalar(z):
	# Test if z is a scalar in regard to matrix multiplication
	return all(not isinstance(z, T) for T in [Matrix, State])


class Matrix(object):

	"""Matrices of arbitrary objects.
	
	In a perfect world, matrices and vectors of Bras and Kets would be
	Numpy arrays with dtype=object. Numpy is sufficiently imperfect that
	it is easier to define an ad hoc Matrix class than to work around it.

	The size of a matrix A is A.ht * A.wd. Its elements are stored in
	column-major order in the list A.elts. Iterating over a row or column
	vector yields its elements.

	Two matrices can be added and multiplied with the usual linear algebra
	operators. They can be multiplied by scalars. A scalar is anything
	except a matrix or a State; States should always be held in Bras or
	Kets. Matrices are array-like for Numpy purposes, and Numpy's linear
	algebra operations can be applied to a Matrix of numbers as is. Binary
	operations are unlikely to work.

	There are no 1*1 Matrices - the element is returned as a scalar
	instead. This is useful more often than it gets in the way.

	Following the convention of Trefethen & Bau, the rows of a matrix are
	complex conjugated. This is almost always what you want.

	"""

	def __new__(cls, m, n, elts):
		if m*n == 1:
			return iter(elts).next()
		else:
			return object.__new__(cls, m, n, elts)

	def __init__(self, m, n, elts):
		"""Return an m*n matrix, containing elts in column-major order."""
		elts = list(elts)
		assert len(elts) == m*n
		self.ht, self.wd, self.elts = m, n, elts
		
	def __repr__(self):
		if self.wd == 1:
			return "col(" +  "; ".join(repr(e) for e in self.elts) + ")"
		elif self.ht == 1:
			return "row(" +  ", ".join(repr(e) for e in self.elts) + ")"
		else:
			return "Matrix(%d, %d, [" % (self.ht, self.wd) + \
				";   ".join(", ".join(repr(e) for e in r) for r in self._rws()) + \
				"])"
		
	def cols(self):
		"""Return the columns as an iterator over self.ht*1 matrices."""
		return (col(x) for x in self._cols())

	def _cols(self):
		for n in xrange(self.wd):
			yield self.elts[n*self.ht:(n+1)*self.ht]
				
	def rows(self):
		"""Return the conjugated rows as an iterator over 1*self.wd matrices."""
		return (row(x) for x in self._rows())
		
	def _rows(self):
		return ([z.conjugate() for z in r] for r in self._rws())
			
	def _rws(self):
		for n in xrange(self.ht):
			yield self.elts[n::self.ht]
			
	def __iter__(self):
		assert self.ht == 1 or self.wd == 1
		return iter(self.elts)
			
	def indices(self):
		"""Return an iterator of (row, column) pairs, in column-major order.
"""
		for n in xrange(self.wd):
			for m in xrange(self.ht):
				yield m, n
			
	def __add__(self, other):
		assert isinstance(other, Matrix)
		assert self.ht is other.ht
		assert self.wd is other.wd
		return Matrix(self.ht, other.wd, (x+y for x,y in zip(self.elts, other.elts)))

	def __sub__(self, other):
		return self + (-1*other)
			
	def __mul__(self, other):
		if isinstance(other, Matrix):
			return self._matmul(other)
		elif _matrix_scalar(other):
			return self._mulsca(other)
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		if _matrix_scalar(other):
			return self._scamul(other)
		else:
			return NotImplemented
			
	def __div__(self, other):
		assert _matrix_scalar(other)
		return self._mulsca(1./other)

	def __array__(self, dtype=complex):
		assert all(isinstance(z, numbers.Complex) for z in self.elts)
		return numpy.array([r for r in self._rws()], dtype)
			
	def _mulsca(self, z):
		return Matrix(self.ht, self.wd, (x*z for x in self.elts))
			
	def _scamul(self, z):
		return Matrix(self.ht, self.wd, (z*x for x in self.elts))
			
	def _matmul(self, other):
		assert self.wd is other.ht
		# the conjugate in dot reverses the one in rows
		return Matrix(self.ht, other.wd,
			(dot(r, c) for c in other._cols() for r in self._rows()))
			
	def conjugate(self):
		"""Return the Hermitian conjugate."""
		return Matrix(self.wd, self.ht,
			(z for r in self._rows() for z in r))

	def map(self, f):
		return Matrix(self.ht, self.wd, (f(x) for x in self.elts))
		
			
# bras and kets

def dirac_scalar(z):
	return all(not isinstance(z, T) for T in [Matrix, State, Operator, Braket])
			
class Braket(object):
	def __init__(self, s):
		assert isinstance(s, State)
		self.s = s
		
	def __add__(self, other):
		assert isinstance(other, type(self))
		return type(self)(self.s+other.s)

	def __sub__(self, other):
		return self + (-1*other)
		
	def __div__(self,other):
		return self*(1./other)
				
			
class Bra(Braket):
	def __repr__(self):
		return "Bra(" + repr(self.s) + ")"

	def __mul__(self, other):
		if isinstance(other, Ket):
			return self.s * other.s
		elif dirac_scalar(other):
			# The conjugate below is the one that Dirac notation
			# cleverly hides. It appears in <alpha|a^\dagger =
			# alpha^*<alpha|, for instance. 
			return Bra(self.s.scaled(other.conjugate()))
		else:
			return NotImplemented

	def __rmul__(self, other):
		if dirac_scalar(other):
			return type(self)(self.s.scaled(other.conjugate()))
		else:
			return NotImplemented
			
	# D() and C() will be implemented when necessary.
		
	def conjugate(self):
		return Ket(self.s)
	
		
class Ket(Braket):
	def __repr__(self):
		return "Ket(" + repr(self.s) + ")"

	def __mul__(self, other):
		if dirac_scalar(other):
			return type(self)(self.s.scaled(other))
		else:
			return NotImplemented

	def __rmul__(self, other):
		if dirac_scalar(other):
			return type(self)(self.s.scaled(other))
		else:
			return NotImplemented
		
	def conjugate(self):
		return Bra(self.s)
		
	def prms(self):
		"""Return the parameters of the state.

		 This works for coherent states and superpositions of them,
		 which are images of the representation function. The
		 parameters of a Ket are returned in a column, in order to
		 multiply Jacobians on the right.
		 """
		return col(self.s.prms())
		
	def smrp(self, zs):
		"""Return a state reconstituted from parameters.

		 zs is a column of parameters, similar to that returned by
		 self.prms(). This works for coherent states and
		 superpositions of them, which are images of the
		 representation function.
		 """
		assert isinstance(zs, Matrix) and zs.wd == 1
		return Ket(self.s.smrp(zs.elts))
		
	def D(self):
		"""Return the direct Wirtinger derivative.

		If prms is defined for a Ket Q, then
			Q.smrp(q.prms() + h) ~
				Q + Q.D()*h + h.conjugate()*Q.C()
		In order for this to be defined, the D() method returns a row
		of kets.
		"""
		return row(Ket(s) for s in self.s.D())
		
	def C(self):
		"""Return the conjugate Wirtinger derivative.

		If prms is defined for a Ket Q, then
			Q.smrp(q.prms() + h) ~
				Q + Q.D()*h + h.conjugate()*Q.C()
		In order for this to be defined, the C() method returns a
		column of kets. That sounds odd, but the kets have always been
		zero so far.
		"""
		return col([Ket(FockExpansion(0))]*len(self.s.prms()))
			

##################################################
#
#	operators
#
##################################################

class Operator(object):
	def __init__(self, bfun, kfun):
		"""Construct a Hilbert space operator.

		The function bfun applies the operator to a bra, while kfun
		applies it to a ket."""
		self.bfun = bfun
		self.kfun = kfun

	def __sub__(self, other):
		return self + (-1*other)
		
	def __add__(self, other):
		if isinstance(other, Operator):
			return Operator(lambda b: self.bfun(b) + other.bfun(b),
				lambda k: self.kfun(k) + other.kfun(k))
		else:
			return NotImplemented
			
	def __mul__(self, other):
		if isinstance(other, Operator):
			return Operator(lambda b: other.bfun(self.bfun(b)),
				lambda k: self.kfun(other.kfun(k)))
		elif isinstance(other, Ket):
			return Ket(self.kfun(other.s))
		elif dirac_scalar(other):
			return Operator(lambda b: self.bfun(b).scaled(other.conjugate()),
				lambda k: self.kfun(k).scaled(other))
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		if isinstance(other, Bra):
			return Bra(self.bfun(other.s))
		elif dirac_scalar(other):
			return self*other
		else:
			return NotImplemented
			
	def conjugate(self):
		return Operator(self.kfun, self.bfun)

			
lop = Operator(lambda b: b.raised(), lambda k: k.lowered())
"""The lowering operator"""
			

##################################################
#
#	Fock states
#
##################################################

def madd(*terms):
	return sum((0 if x is None else x) for x in terms)

class State(object):
	pass

class FockExpansion(State):
	def __init__(self, *cs):
		self.cs = tuple(complex(c) for c in cs)
		
	def __repr__(self):
		return "FockExpansion(" + \
			", ".join((repr(c) if c else "0") for c in self.cs) + ")"
		
	def __add__(self, other):
		if isinstance(other, FockExpansion):
			return FockExpansion(*(madd(z,w) for z, w in map(None, self.cs, other.cs)))
		else:
			return NotImplemented
		
	def __mul__(self, other):
		if isinstance(other, FockExpansion):
			return dot(self.cs, other.cs)
		else:
			return NotImplemented
			
	def raised(self):
		rcs = [sqrt(n+1)*self.cs[n] for n in xrange(len(self.cs))]
		return FockExpansion(*[0]+rcs)
		
	def lowered(self):
		lcs = [sqrt(n)*self.cs[n] for n in xrange(1,len(self.cs))]
		return FockExpansion(*lcs)
	
	def scaled(self, z):
		return FockExpansion(*(c*z for c in self.cs))
		
	def isvac(self):
		return all(z == 0 for z in self.cs[1:])
			

##################################################
#
#	displaced Fock expansions
#
##################################################

class DisplacedState(State):
	"state s, displaced by coherent amplitude a"
	def __init__(self, a, s):
		self.s = s
		self.a = complex(a)

	def __repr__(self):
		return "DisplacedState(%s, %s)" % (repr(self.a), repr(self.s))
		
	def __add__(self, other):
		if isinstance(other, DisplacedState) and other.a == self.a:
			return DisplacedState(self.a, self.s + other.s)
		elif isinstance(other, DisplacedState):
			return Sum(self, other)
		elif isinstance(other, FockExpansion):
			return self + DisplacedState(0, other)
		else:
			return NotImplemented
			
	def __radd__(self, other):
		if isinstance(other, FockExpansion):
			return self + other
		else:
			return NotImplemented
			
	def __mul__(self, other):
		if isinstance(other, DisplacedState):
			return ddmul(self, other)
		elif isinstance(other, FockExpansion):
			return ddmul(self, DisplacedState(0, other))
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		if isinstance(other, FockExpansion):
			return ddmul(DisplacedState(0, other), self)
		else:
			return NotImplemented
			
	def scaled(self, z):
		return DisplacedState(self.a, self.s.scaled(z))
		
	def lowered(self):
		return DisplacedState(self.a, self.s.lowered() + self.s.scaled(self.a))
		
	def raised(self):
		return DisplacedState(self.a, self.s.raised() + self.s.scaled(self.a.conjugate()))
		
	def prms(self):
		assert self.s.isvac()
		return (-0.5*abs(self.a)**2 + log(FockExpansion(1)*self.s), self.a)
		
	def smrp(self, zs):
		assert self.s.isvac()
		return DisplacedState(zs[1], FockExpansion(exp(zs[0]+0.5*abs(zs[1])**2)))
		
	def D(self):
		return (self, self.raised())
			
def ddmul(bra, ket):
	# see soften.tex
	b, a = bra.a, ket.a
	return  exp(-0.5*abs(a)**2-0.5*abs(b)**2+b.conjugate()*a) * (
		explower((a-b).conjugate(), bra.s) *
		explower(-(a-b).conjugate(), ket.s))
		
def explower(z, s):
	"""Return exp(z*lop) applied to |s>."""
	# ensure exp(a)|s> is a finite polynomial
	assert isinstance(s, FockExpansion)
	sm, term, n = s, s.lowered().scaled(z), 1
	while abs(term*term) > 0:
		sm += term
		n += 1
		term = term.lowered().scaled(z/n)
	return sm
			

##################################################
#
#	sums of displaced states
#
##################################################

class Sum(State):
	# the new idea is that construction takes the terms as given.
	# the condition method, yet to be written, will combine terms
	# with close amplitudes into displaced Fock expansions.
	def __init__(self, *ts):
		self.ts = []		# terms
		for s in ts:
			if isinstance(s, FockExpansion):
				s = DisplacedState(0, s)
			assert isinstance(s, DisplacedState)
			self.ts.append(s)
				
	def terms(self):
		return self.ts
				
	def __repr__(self):
		return "Sum(" + ", ".join(repr(s) for s in self.terms()) + ")"
		
	def __add__(self, other):
		if isinstance(other, Sum):
			return Sum(*self.terms() + other.terms())
		elif isinstance(other, State):
			return Sum(*self.terms() + [other])
		else:
			return NotImplemented
			
	def __radd__(self, other):
		assert isinstance(other, State)
		return self + other
		
	def scaled(self, z):
		return Sum(*(s.scaled(z) for s in self.terms()))
		
	def __mul__(self, other):
		assert isinstance(other, State)
		if not isinstance(other, Sum):
			other = Sum(other)
		return sum(b*k for b in self.terms() for k in other.terms())
		
	def __rmul__(self, other):
		assert isinstance(other, State)
		if not isinstance(other, Sum):
			other = Sum(other)
		return sum(b*k for b in other.terms() for k in self.terms())
		
	def prms(self):
		return tuple(z for s in self.terms() for z in s.prms())
		
	def smrp(self, zs):
		s = Sum()
		for t in self.terms():
			n = len(t.prms())
			s = s + t.smrp(zs[:n])
			zs = zs[n:]
		return s
		
	def D(self):
		return tuple(t for s in self.terms() for t in s.D())

	def lowered(self):
		return Sum(*[t.lowered() for t in self.terms()])

	def raised(self):
		return Sum(*[t.raised() for t in self.terms()])
	
	
##################################################
#
#	utilities
#
##################################################
		
def norm(q):
	"""Return the 2 norm of a Bra, Ket, or vector."""
	return sqrt(abs(q.conjugate() * q))

def coherent(alpha):
	"""Return a coherent state with amplitude alpha."""
	return DisplacedState(alpha, FockExpansion(1))

def number(n):
	"""Return a Fock state with n particles."""
	return FockExpansion(*[0]*n + [1])