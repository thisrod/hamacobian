# Tests, in the standard unit testing framework.

from unittest import TestCase, main as run_tests
from states import *
from numpy import array, identity, exp, log, sum, sqrt, allclose, zeros
from scipy.misc import factorial
from numpy.linalg import norm

def expands_to(basis, state, expansion):
	return allclose((basis*state).flatten(), expansion)

class CoherentTestCase(TestCase):
	def setUp(self):
		self.alpha = 3+4j
		self.glauber = CoherentRow(0, self.alpha).sum().normalised()
		self.basis = FockRow(*[1]*15)
		self.fock = FockRow(*(self.basis*self.glauber).flatten()).sum()
		n = array(xrange(len(self.basis)))
		self.expansion = exp(-0.5*abs(self.alpha)**2)*self.alpha**n/sqrt(factorial(n))
		
class ExpansionTest(CoherentTestCase):
	"Verify the expansion of glauber() over Fock states"
	def runTest(self):
		assert expands_to(self.basis, self.glauber, self.expansion)
		assert expands_to(self.basis, self.fock, self.expansion)
		
class FirstDerivativeTest(CoherentTestCase):
	"Check that the product of a coherent state with its derivative is the derivative of the product"
	def runTest(self):
		h = 1e-4
		D = self.glauber.D()
		n = len(D)
		E = h*identity(n)
		exact = self.basis*D
		for i in xrange(n):
			approx = ((self.basis*(self.glauber.components+E[i,:]))-(self.basis*(self.glauber.components+(-E[i,:]))))/2/h
			assert allclose(approx.flatten(),exact[:,i])
		
class SecondDerivativeTest(CoherentTestCase):
	"Check that the product of two coherent state derivatives is the derivative of the product of a coherent state with one derivative"
	def runTest(self):
		h = 1e-4
		D = self.glauber.D()
		n = len(D)
		E = h*identity(n)
		exact = D*D
		for i in xrange(n):
			approx = ((D*(self.glauber+E[i,:]))-(D*(self.glauber+(-E[i,:]))))/2/h
			assert allclose(approx.flatten(),exact[:,i])
				
if __name__ == "__main__":
            run_tests()