from cmath import *

##################################################
#
#	matrices of objects
#
##################################################

def row(elts):
	return Matrix(1, len(elts), elts)

def col(elts):
	return Matrix(len(elts), 1, elts)

def dot(xs, ys):
	return sum(x.conjugate()*y for (x, y) in zip(xs, ys))
	
def matrix_scalar(z):
	return all(not isinstance(z, T) for T in [Matrix, State])

class Matrix(object):

	def __init__(self, m, n, elts):
		"elts are in column major order"
		
		self.ht, self.wd, self.elts = m, n, list(elts)
		assert len(self.elts) is m*n
		
	def __repr__(self):
		return "Matrix(%d, %d, [" % (self.ht, self.wd) + \
			", ".join(repr(e) for e in self.elts) + \
			"])"
		
	def cols(self):
		return (self.elts[n*self.ht:(n+1)*self.ht]
			for n in xrange(self.wd))
				
	def rows(self):
		# Trefethen & Bau shows that you usually want conjugated rows
		return ([z.conjugate() for z in self.elts[n::self.ht]]
			for n in xrange(self.ht))
			
	def __add__(self, other):
		assert isinstance(other, type(self))
		assert self.ht is other.ht
		assert self.wd is other.wd
		return Matrix(self.ht, other.wd, (x+y for x in self.elts for y in other.elts))
			
	def __mul__(self, other):
		if isinstance(other, Matrix):
			return self.matmul(other)
		elif matrix_scalar(other):
			return self.mulsca(other)
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		if matrix_scalar(other):
			return self.scamul(other)
		else:
			return NotImplemented
			
	def mulsca(self, z):
		return Matrix(self.ht, self.wd, (x*z for x in self.elts))
			
	def scamul(self, z):
		return Matrix(self.ht, self.wd, (z*x for x in self.elts))
			
	def matmul(self, other):
		assert self.wd is other.ht
		return Matrix(self.ht, other.wd,
			(dot(r, c) for c in other.cols() for r in self.rows()))
			
	def conjugate(self):
		return Matrix(self.wd, self.ht,
			(z for r in self.rows() for z in r))
		
			
##################################################
#
#	bras and kets
#
##################################################

def dirac_scalar(z):
	return all(not isinstance(z, T) for T in [Matrix, State, Operator, Braket])
			
class Braket(object):
	def __init__(self, s):
		assert isinstance(s, State)
		self.s = s
		
	def __add__(self, other):
		assert isinstance(other, type(self))
		return type(self)(self.s+other.s)
				
			
class Bra(Braket):
	def __repr__(self):
		return "Bra(" + repr(self.s) + ")"

	def __mul__(self, other):
		if isinstance(other, Ket):
			return self.s * other.s
		elif dirac_scalar(other):
			return type(self)(self.s.scaled(other.conjugate()))
		else:
			return NotImplemented

	def __rmul__(self, other):
		if dirac_scalar(other):
			return type(self)(self.s.scaled(other.conjugate()))
		else:
			return NotImplemented
		
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
			

##################################################
#
#	operators
#
##################################################

class Operator(object):
	def __init__(self, bfun, kfun):
		"bfun transforms bra states, kfun ket states"
		self.bfun = bfun
		self.kfun = kfun
		
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
			return Operator(lambda b: other*self.bfun(b),
				lambda k: other*self.kfun(k))
		else:
			return NotImplemented
			
	def __rmul__(self, other):
		if isinstance(other, Bra):
			return Bra(self.bfun(other.s))
		elif dirac_scalar(other):
			return Operator(lambda b: other*self.bfun(b),
				lambda k: other*self.kfun(k))
		else:
			return NotImplemented
			
	def conjugate(self):
		return Operator(self.kfun, self.bfun)
			
		
lop = Operator(lambda b: b.raised(), lambda k: k.lowered())
			

##################################################
#
#	Fock states
#
##################################################

class State(object):
	pass

class FockExpansion(State):
	def __init__(self, *cs):
		self.cs = cs
		
	def __repr__(self):
		return "FockExpansion(" + \
			", ".join(repr(c) for c in self.cs) + ")"
		
	def __add__(self, other):
		assert isinstance(other, type(self))
		return FockExpansion(*(z+w for z, w in zip(self.cs, other.cs)))
		
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
			

##################################################
#
#	coherent states and similar
#
##################################################

class DisplacedState(State):
	def __init__(self, s, a):
		assert isinstance(s, FockExpansion)
		self.s = s
		self.a = a