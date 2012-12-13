"""
Partial evaluation of matrices that comprise blocks of polynomials.

"""

import numpy as np
from string import ascii_lowercase
import numbers

dvars = ascii_lowercase[-3:] + ascii_lowercase[-4:13:-1] + ascii_lowercase[:8]
"""Letters to represent variables when printing forms."""

class Form(object):

	"""Mathematical formulae in several variables.

	A Form has a subst() method, which takes ndarrays of numbers
	as arguments, and returns the values of the formula when those
	numbers are substituted for the variables.
	"""

	def similar(self, other):
		return self.shape() == other.shape()
		
	def valid(self, *xss):
		"""Test if parameters have the right shapes to substitute."""
		return all(n == xs.shape[1] for n, xs in zip(self.shape(), xss))
		
	def __mul__(self, other):
		if isinstance(other, numbers.Complex):
			return LinearCombination((other, self))
		else:
			return NotImplemented
		
	def __rmul__(self, other):
		if isinstance(other, numbers.Complex):
			return LinearCombination((other, self))
		else:
			return NotImplemented
			
	def __ne__(self, other):
		return not self == other
			
	def __add__(self, other):
		return LinearCombination((1,self))+other
			
	def __neg__(self):
		return -1*self
		
	def __sub__(self,other):
		return self+(-other)


class ScalarForm(Form):

	"""A formula that evaluates to a single object.
	"""

	def constant(self, z):
		"""Return a form similar to self, that evaluates to the constant z.
		"""
		return LinearCombination((z, Monomial(*[0]*self.shape()[0])))
	

class CNumForm(ScalarForm):

	"""A formula that evaluates to a complex number.
	"""
	
	pass
	
	
class StateForm(ScalarForm):
	
	"""A formula that evaluates to a bra or ket."""
	
	pass
	
	
class OpForm(ScalarForm):

	"""A formula for a quantum operator."""
	
	pass
	

class MatrixForm(Form):

	"""A form that substitutes several sets of variables, and collects the results in a matrix."""
	
	pass


class Monomial(CNumForm):

	"""A scalar pattern to evaluate a monomial in several variables.
	
	Bugs: Monomials must have at least one variable.
	"""

	def __init__(self, *ns):
		"""Construct a monomial x[i]**ns[i]."""
		self.ns = ns
		
	def shape(self):
		return (len(self.ns),)

	def subst(self, xs):
		assert self.valid(xs)
		ns = np.array(self.ns)[np.newaxis, :]
		return np.product(xs**ns, axis=1)
		
	def __repr__(self):
		return "Monomial(" + ", ".join("%d" % n for n in self.ns) + ")"
			
	def __str__(self, vars = dvars):
		factors = []
		for n, v in zip(self.ns, vars):
			if n == 1:
				factors.append(v)
			elif n != 0:
				factors.append("%s^%d" % (v, n))
		if not factors:
			factors.append("1")
		return " ".join(factors)
		
	def __eq__(self, other):
		if other == 1:
			return all(n == 0 for n in self.ns)
		elif isinstance(other, Monomial):
			return self.ns == other.ns
		else:
			return NotImplemented
		
	def __mul__(self, other):
		"""Identify variables in self with corresponding ones in other."""
		if isinstance(other, Monomial):
			assert self.similar(other)
			return Monomial(*[n+m for n, m in zip(self.ns,other.ns)])
		else:
			return Form.__mul__(self, other)
			
	def mvmul(self, other):
		if isinstance(other, Monomial):
			return Monomial(*self.ns+other.ns)
		else:
			return other.mvrmul(self)
		
		
class LinearCombination(CNumForm):

	"""Linear combinations are flat.  The associative law is applied to combinations of combinations.
	"""

	def __init__(self, *args):
		"""Return a linear combination.

		The args are (coefficient, term) tuples.
		The coefficients should be fixed numbers, not patterns: I
		represent things like polynomials, not G^R ensembles.
		
		Bug: The treatment of zero coefficients is not defined.
		"""
		self.cs = []
		self.ts = []
		for c, t in args:
			if isinstance(t, LinearCombination):
				self.cs += [c*d for d in t.cs]
				self.ts += t.ts
			else:
				self.cs.append(c)
				self.ts.append(t)
		assert not self.ts or all(x.shape() == self.ts[0].shape() for x in self.ts)

	def __repr__(self):
		return "LinearCombination(" + ", ".join("(%s, %s)" % (repr(c), repr(t)) for c, t in zip(self.cs, self.ts)) + ")"
		
	def __str__(self, vars = dvars):
		terms = []
		for c, t in zip(self.cs, self.ts):
			cstr, tstr = repr(c), t.__str__(vars)
			if t == 1:
				terms += ["+", cstr]
			elif c == 1:
				terms += ["+", tstr]
			elif c == -1:
				terms += ["-", tstr]
			elif c.real < 0 and c.imag <= 0:
				# avoid double minus signs
				terms += ["-", "%s %s" % (repr(-c), tstr)]
			elif c != 0:
				terms += ["+", "%s %s" % (cstr, tstr)]
		if not terms:
			terms.append("0")
		if terms[0] == "+":
			terms = terms[1:]
		return " ".join(terms)
		
	def shape(self):
		return self.ts[0].shape()
		
	def subst(self, *vars):
		return np.sum(np.array([c*t.subst(*vars) for c, t in zip(self.cs, self.ts)]), axis=0)
		
	def __add__(self, other):
		if isinstance(other, LinearCombination):
			return LinearCombination(*zip(self.cs, self.ts) + zip(other.cs, other.ts))
		else:
			return self + LinearCombination((1,other))
		
	def __mul__(self, other):
		if isinstance(other, numbers.Complex):
			return LinearCombination(*zip((other*c for c in self.cs), self.ts))
		else:
			return LinearCombination(*zip(self.cs, (t*other for t in self.ts)))
		
	def __rmul__(self, other):
		if isinstance(other, numbers.Complex):
			return LinearCombination(*zip((other*c for c in self.cs), self.ts))
		else:
			return LinearCombination(*zip(self.cs, (other*t for t in self.ts)))
			
	def mvmul(self, other):
		return LinearCombination(*zip(self.cs, (t.mvmul(other) for t in self.ts)))
			
	def mvrmul(self, other):
		return LinearCombination(*zip(self.cs, (other.mvmul(t) for t in self.ts)))
			
	def collect(self):
		a = []
		cts = zip(self.cs, self.ts)
		while cts:
			t = cts[0][1]
			xs = [ct for ct in cts if ct[1]==t]
			a.append((sum(x[0] for x in xs), t))
			cts = [ct for ct in cts if ct[1]!=t]
		return LinearCombination(*a)
		
		
class FockExpansion(StateForm):
	
	def __init__(self, *cs):
		"""Return a state with the given amplitudes for number states.
		"""
		self.cs = cs
		
	def __repr__(self):
		return "FockExpansion(" + ", ".join(repr(c) for c in self.cs) + ")"
		
	def __mul__(self, other):
		# conjugated parameters are substituted into bras
		s = self.cs[0].mvmul(other.cs[0])
		cs = self.cs[1:]
		ds = other.cs[1:]
		while cs and ds:
			s += cs[0].mvmul(ds[0])
			cs = cs[1:]
			ds = ds[1:]
		return s
		
	def lowered(self):
		p = len(self.cs)
		return FockExpansion(*[np.sqrt(n)*c for n, c in zip(xrange(1,p), self.cs[1:])])
		
	def raised(self):
		pass
		p = len(self.cs)
		return FockExpansion(*[self.cs[0].constant(0)]+[np.sqrt(n+1)*c for n, c in zip(xrange(p), self.cs)])


class Row(MatrixForm):
	
	"""A matrix, whose elements recur for each set of arguments.
	"""
	
	def __init__(self, *elts):
		self.elts = elts
		
	def __repr__(self):
		return  "Row(" + ", ".join(repr(c) for c in self.elts) + ")"
		
	def subst(self, xs):
		m, n = len(self.elts), xs.shape[0]
		ys = np.empty((m*n,), dtype=complex)
		for i in xrange(m):
			ys[i::n] = self.elts[i].subst(xs)
		return ys