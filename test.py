#
# Functions to test matrix initialisation functions.
# For now, these are single mode only.
#

from numpy import ndarray, array

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
	"""The result is not Hermitian.  Actual is a set of indices of elements that aren't the conjugate of their opposites.  The assume the matrix is square."""
	@classmethod
	def exclude(cls):
		A = cls.output
		goats = [(j,k) for j in xrange(A.shape[0]) for k in xrange(j+1) if A[j,k] != A[k,j].conjugate()]
		if goats: raise cls(goats, [])
	
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
	
	
#
# Test stub functions that should fail in particular ways
#

if __name__ =="__main__":

	prefix = "Tests failed to detect "
	z1 = array([[0, 2]])	# Coherent state, amplitude 2.
	z2 = array([[0, 2],[0, 2]])	# Two identical components.

	try: test_rho(lambda z: None, None, None, z1)
	except NotAMatrix: pass
	else:	raise Exception(prefix + "a non-array rho")

	try: test_rho(lambda z: z, None, None, z1)
	except WrongShape: pass
	else:	raise Exception(prefix + "a rho with the wrong shape")

	try: test_rho(lambda z: array([[1j]]), None, None, z1)
	except Nonhermitian: pass
	else:	raise Exception(prefix + "a non-hermitian rho")

	try: test_rho(lambda z: array([[1,2],[2,1]]), None, None, z2)
	except NotCatenation: pass
	else:	raise Exception(prefix + "different moments between identical states")
