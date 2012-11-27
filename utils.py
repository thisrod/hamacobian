"""General utility code for hamacobian"""

from copy import copy
import numpy as np

# Absolute value dispatch for patterns

def abs(z):
	if isinstance(z, Pattern):
		return ElementalPattern(z, np.abs)
	else:
		return np.abs(z)

def randc():
	"""Return a complex number drawn from a normal distribution."""
	return gauss(0,1) + gauss(0,1)*1j

class Coefficients(dict):
	"""A sequence of coefficients.
	
	Can be added to other coefficients, and multiplied by scalars.  This is a candidate for writing in C.
	"""
	
	def __init__(self, cs, zero=0+0j):
		"""The final coefficient must be zero."""
		self.n = zero
		for i in xrange(len(cs)):
			self[i] = cs[i]
		self.reduce()
		
	def __getitem__(self, i):
		if dict.__contains__(self, i):
			return dict.__getitem__(self, i)
		else:
			return self.n
		
	def __mul__(self, x):
		"""Greedy."""
		return Coefficients([c*x for c in self.values()], self.n)
		
	def __rmul__(self, x):
		return self*x
		
	def __add__(self, other):
		if not isinstance(other, Coefficients):
			return NotImplemented
		assert self.n == other.n
		sum = copy(self)
		for i, c in other.iteritems():
			sum[i] = sum[i] + c
		sum.reduce()
		return sum
		
	def reduce(self):
		"""Remove zero coefficients"""
		for i in self.keys():
			if self[i] == self.n:
				del self[i]