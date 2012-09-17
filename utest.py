# Tests, in the standard unit testing framework.

from unittest import TestCase, main as run_tests
from naive import *
from numpy import array, exp, log, sum, sqrt, allclose, zeros
from scipy.misc import factorial

def lnorm(z):
	return 0.5*log(sum(rho(z,z)))

class AmplitudeTwoCase(TestCase):
	"Set up a normal coherent state, with  real amplitude 2"
	def setUp(self):
		self.zs = array([[0, 2+0j]])
		self.zs[0,0] -= lnorm(self.zs)
		assert lnorm(self.zs) == 0
		
		
class FockTest(AmplitudeTwoCase):
	def runTest(self):
		"Verify the expansion of |2+0j> over Fock states."
		alpha = 2
		for n in xrange(10):
			cs = zeros(n+1);  cs[-1] = 1
			assert allclose(jfock(self.zs, cs)[0,0],
				exp(-0.5*abs(alpha)**2)*alpha.conjugate()**n/sqrt(factorial(n)))
				
if __name__ == "__main__":
            run_tests()