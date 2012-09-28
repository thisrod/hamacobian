# Tests, in the standard unit testing framework.

from unittest import TestCase, main as run_tests
from states import *
from numpy import array, identity, exp, log, sum, dot, sqrt, allclose, newaxis, zeros
from numpy.random import randn
from numpy.linalg import norm
from scipy.misc import factorial

def randc(n):
	return dot(randn(n,2), [1,1j])
	
	

class NumberCase(TestCase):
	def setUp(self):
		self.basis = FockStates(*[1]*5).bras()
		self.two = FockStates(0, 0, 1).sum()
		self.z = randc(1)[0]

class SingleModeCase(TestCase):
	def setUp(self):
		self.basis = FockStates(*[1]*20).bras()
		self.z = randc(1)[0]
		self.samples = CoherentStates(0,0, 0,1.8, *array(zip(randn(3), randc(3))).flatten())
		for ket in self.samples.kets():
			ket /= sqrt(ket.H()*ket)
			
			
			
class ScalarsAssociate(NumberCase):
	def runTest(self):
		z, two, basis = self.z, self.two, self.basis
		assert allclose((z*basis)*two, z*(basis*two))
		assert allclose((basis*z)*two, basis*(z*two))
		assert allclose((basis*two)*z, basis*(two*z))
		
class ScalarsCommute(NumberCase):
	def runTest(self):
		z, two, basis = self.z, self.two, self.basis
		assert allclose(z*basis*two, basis*z*two)
		assert allclose(z*basis*two, basis*two*z)
		
class ExpansionTest(SingleModeCase):
	def runTest(self):
		for ket in self.samples.kets():
			a = ket.a
			cs = (self.basis * ket).flatten()
			for n in xrange(len(self.basis)):
				assert allclose(cs[n], exp(-0.5*abs(a)**2) * a**n / sqrt(factorial(n)) )
				
class NumberTest(NumberCase):
	def runTest(self):
		assert allclose(2, self.two.H()*self.two.lowered().raised())
		
class CoherentStateBracketTest(SingleModeCase):
	"Verify the inner products of coherent states"
	def runTest(self):
		for bra in self.samples.bras():
			for ket in self.samples.kets():
				assert allclose(bra*ket, exp(-0.5*abs(bra.a)**2 -0.5*abs(ket.a)**2 + bra.a.conj()*ket.a))
				
class ApproximateDerivativeTest(SingleModeCase):
	def runTest(self):
		basis = self.basis
		hs = self.z*1e-4*eye(2)
		for ket in self.samples.kets():
			for h in hs:
				assert allclose(basis*(ket+h) - basis*(ket-h), dot(basis*ket.D(), 2*h))

class CoherentStateScalingTest(SingleModeCase):
	"Verify that norms of coherent states are linear in scalar multiplication"
	def runTest(self):
		for ket in self.samples.kets():
			assert allclose(self.basis*(self.z*ket), self.z*(self.basis*ket))



if __name__ == "__main__":
            run_tests()