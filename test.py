#
# Functions to test matrix initialisation functions.
# For now, these are single mode only.
#

from numpy import ndarray

class Failure(Exception):
	"""A test with valid inputs produced an incorrect result.
	
	Subclasses define an exclude method, which raises the exception if appropriate.  This might be a class method, if there's no need to construct an instance that won't be raised."""
	
	@classmethod
	def set_up(cls, operation, z):
		"Assert that z is a valid ensemble, and prepare to test operation."
		assert isinstance(z, ndarray)
		assert z.ndim == 2
		n, m = z.shape
		assert m == 2	# Single mode implementation restriction
		cls.operation = operation
		cls.input = z.copy()
		cls.output = operation(z)
		
class NotAMatrix(Failure):
	"A function that should have returned an ndarray returned something else."
	@classmethod
	def exclude(cls):
		if not isinstance(cls.output, ndarray): raise cls

class WrongShape(Failure):
	"The shape of a calculated ndarray was inconsistent with the shape of the input."
	@classmethod
	def exclude(cls, *s):
		if cls.output.shape != s: raise cls
		
class Nonhermitian(Failure):
	"""The result is not Hermitian.  Actual is a set of indices of elements that aren't the conjugate of their opposites."""
	@classmethod
	def exclude(cls):
		pass
	
class NotCatenation(Failure):
	@classmethod
	def exclude(cls):
		pass


def test_rho(rho, jsq, jham, z):
	Failure.set_up(rho, z)
	n, m = z.shape
	NotAMatrix.exclude()
	Nonhermitian.exclude()
	WrongShape.exclude(n,n)
	NotCatenation.exclude()
	
	
def test_jsq(rho, jsq, jham, z):
	n, m = Failure.set_up(jsq, z)
	NotAMatrix.exclude()
	Nonhermitian.exclude()
	WrongShape.exclude(n*m,n*m)
	