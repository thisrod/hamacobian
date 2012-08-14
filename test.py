#
# Functions to test matrix initialisation functions.
# For now, these are single mode only.
#

from numpy import ndarray

class Failure(Exception):
	"""A test with valid inputs produced an incorrect result.
	
	Subclasses define an exclude method, which raises the exception if appropriate.  This might be a class method, if there's no need to construct an instance that won't be raised."""
	
	@classmethod
	set_up(class, operation, z):
		"""Assert that z is a valid ensemble, and prepare to test operation.
		Return the number of components and modes+1.
		The result is stored in the class variable Failure.result"""
		
		assert isinstance(z, ndarray)
		assert z.ndim == 2
		n, m = z.shape
		assert m == 2	# Single mode implementation restriction
		class.operation = operation
		class.ensemble = z.copy()
		class.result = operation(z)
		return n, m
		
class FailureToRelate(Failure):
	"""Test that some property of the result equals the value supplied on construction.
	Subclasses should override the property() method, which defaults to the identity."""	
	
	def __init__(self, expected):
		self.expected = expected
	
	def exclude(self):
		self.actual = self.property(result)
		self.relates(self.actual, self.expected) or raise self
		
	def property(self, result):
		result
		
class FailureToEqual(FailureToRelate):
	def relates(self, actual, expected):
		actual == expected
		
class FailureToBe(FailureToRelate):
	def relates(self, actual, expected):
		actual is expected
		

class NotAMatrix(FailureToBe):
	"A function that should have returned an ndarray returned something else."
	def __init__(self):
		self.expected = ndarray
	def property(self, result):
		type(result)

class WrongShape(FailureToEqual):
	"The shape of a calculated ndarray was inconsistent with the shape of the input."
	def __init__(self, *s):
		self.expected = s
	def property(self, result):
		result.shape
		
class Nonhermitian(Failure):
	"""The result is not Hermitian.  Actual is a set of indices of elements that aren't the conjugate of their opposites."""
	pass


def test_rho(rho, jsq, jham, z):
	n, m = Failure.set_up(rho, z)
	NotAMatrix.exclude()
	Nonhermitian.exclude()
	WrongShape.exclude(n,n)
	
	
def test_jsq(rho, jsq, jham, z):
	n, m = Failure.set_up(jsq, z)
	NotAMatrix.exclude()
	Nonhermitian.exclude()
	WrongShape.exclude(n*m,n*m)
	