"""
Partial evaluation of matrices that comprise blocks of polynomials.

The module "states" evaluates expressions of bras, kets, operators, and matrices of them.  This evaluation is complex, and should be avoided in inner loops.  This module defines a class Pattern, and various subclasses, which are closed under the arithmetic operations required by states, so that the classes in states can be instantiated with pattern variables instead of complex numbers.  An expression of these objects evaluates to a pattern in the variables: this evaluation can be done once, and numbers substituted for the variables inside a loop.

Classes:
------

DoubleMonomialPattern:  
"""

import numpy as np
from utils import *
from string import ascii_lowercase

dvars = ascii_lowercase[-3:] + ascii_lowercase[-4:13:-1] + ascii_lowercase[:8]

class Pattern(object):
	"""Supertype for dispatch"""
	pass

class MonomialPattern(Pattern, Coefficients):
	"""A textbook monomial, in many variables.  Takes one set of values, and returns a 1D array.
	
	The constructor takes a list of exponents, inherited from Coefficients.  Coefficient addition becomes Monomial multiplication.
	
	Note that a unit monomial is false for logical purposes.
	"""
		
	def subst(self, xs):
		result = np.ones(xs.shape[0], xs.dtype)
		for i, p in self.iteritems():
			result *= xs[:, i]**int(p.real)
		return result
			
	def __str__(self, vars = dvars):
		v = self.keys()
		v.sort()
		terms = []
		for i in v:
			if self[i] == 1:
				terms.append(vars[i])
			else:
				terms.append("%s^%d" % (vars[i], self[i]))
		if not terms:
			terms.append("1")
		return " ".join(terms)
			
			
	def __add__(self, other):
		return NotImplemented
		
	def __mul__(self, other):
		if isinstance(other, MonomialPattern):
			return Coefficients.__add__(self, other)
		else:
			return NotImplemented
		

class DoubleMonomialPattern(Pattern):
	"""Raise the conjugated left variables, and the right variables, to powers."""

	def __init__(self, *parts):
		"""parts are lc, and r as lists of powers"""
		self.parts = [MonomialPattern(p) for p in parts]
		
	def __str__(self, lvars=dvars, rvars=dvars, const=1):	
		lvars = [x+'*' for x in lvars]
		if const == 1:
			c = ''
		else:
			c = repr(const) + ' '
		l, r = self.parts[0].__str__(lvars), self.parts[1].__str__(rvars)
		if not (self.parts[0] or self.parts[1]):
			return repr(const)
		elif not self.parts[0]:
			return c+r
		elif not self.parts[1]:
			return c+l
		else:
			return c+ l + " " + r
			
	def __eq__(self, other):
		return isinstance(other, DoubleMonomialPattern) and all(x == y for x,y in zip(self.parts, other.parts))
				
	def subst(self, ls, rs):
		"""First dimension of ls and rs selects the components, second dimension the variables"""
		ls = np.array(ls)
		rs = np.array(rs)
		return np.outer(self.parts[0].subst(ls.conj()), self.parts[1].subst(rs))
		
	def __mul__(self, other):
		if isinstance(other, DoubleMonomialPattern):
			product = DoubleMonomialPattern()
			product.parts = [p*q for p, q in zip(self.parts, other.parts)]
			return product
		else:
			return NotImplemented
		
	def __rmul__(self, other):
		return self*other
		
	def __add__(self, other):
		return self.asPolynomial() + other
		
	def asPolynomial(self):
		return DoublePolynomialPattern([1], [self])
		


class DoublePolynomialPattern(Pattern):

	"""A polynomial in conjugated lvars, and rvars.

	This could be done as a hash table instead of an association list.  That would require monomial patterns to be hashable.  Only subst should run in inner loops, so the slowdown only occurs once.  For now it isn't worth doing.
	"""
	
	def __init__(self, cs, ds):
		"""the cs are coefficients, and the ds monomials."""
		
		self.cs = Coefficients(cs)
		self.terms = ds
		
	def __str__(self, lvars=dvars, rvars=dvars):
		if not self.terms:
			return "0"
		else:
			return " + ".join(self.terms[i].__str__(lvars, rvars, self.cs[i]) for i in xrange(len(self.terms)))
			
	def __add__(self, other):
		if isinstance(other, DoubleMonomialPattern):
			other = other.asPolynomial()
		if not isinstance(other, DoublePolynomialPattern):
			return NotImplemented
		m,n = len(self.terms), len(other.terms)
		cs = Coefficients([self.cs[i] for i in xrange(m)] + [other.cs[i] for i in xrange(n)])
		sum = DoublePolynomialPattern(cs, self.terms+other.terms)
		return sum

	def __mul__(self, other):
		if isinstance(other, DoubleMonomialPattern):
			other = other.asPolynomial()
		if isinstance(other, DoublePolynomialPattern):
			n = len(self.terms)
			m = len(other.terms)
			cs = [self.cs[i]*other.cs[j] for i in xrange(n) for j in xrange(m)]
			ds = [self.terms[i]*other.terms[j] for i in xrange(n) for j in xrange(m)]
			return DoublePolynomialPattern(Coefficients(cs), ds)
		else:		# scalar
			return DoublePolynomialPattern(other*self.cs, self.terms)
		
	def collect(self):
		ts = [(self.terms[i], self.cs[i]) for i in xrange(len(self.terms))]
		self.terms = []
		self.cs = Coefficients()
		while ts:
			self.terms.append(ts[0][0])
			
			ts = [ts[i] for i in leave]
			cs = [cs[i] for i in leave]
		assert False
			
		
	def asPolynomial(self):
		return self
		
	
class ElementalPattern(Pattern):
	"""Substitute into a base pattern, then apply a numpy ufun."""
	
	def __init__(self, pattern, f):
		self.base = pattern
		self.ufun = f
		
	def subst(self, *args):
		return self.ufun(self.base.subst(*args))
	

def lvar(i):
	"""Left variable number i"""
	return DoubleMonomialPattern([0]*i+[1], [])

def rvar(i):
	"""Conjugate of left variable number i"""
	return DoubleMonomialPattern([], [0]*i+[1])
