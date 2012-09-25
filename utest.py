# Tests, in the standard unit testing framework.

from unittest import TestCase, main as run_tests
from states import *
from numpy import array, exp, log, sum, sqrt, allclose, zeros
from scipy.misc import factorial

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
	def runTest(self):
		"Verify the expansion of glauber() over Fock states"
		assert expands_to(self.basis, self.glauber, self.expansion)
		assert expands_to(self.basis, self.fock, self.expansion)
		
class DerivativeTest(CoherentTestCase):
	pass
				
if __name__ == "__main__":
            run_tests()