# Tests, in the standard unit testing framework.

from unittest import TestCase, main as run_tests
from states import *
from numpy import array, identity, exp, log, sum, dot, sqrt, allclose, newaxis, zeros, outer
from numpy.random import randn
from numpy.linalg import norm
from scipy.misc import factorial

def randc(n):
	return dot(randn(n,2), [1,1j])
	
def closenough(z, w):
	return allclose(array(z, dtype=complex), array(w, dtype=complex))
	
vac = FockExpansion([1])
	

class NumberCase(TestCase):
	def setUp(self):
		self.basis = array([FockExpansion(e) for e in eye(7)])
		self.two = FockExpansion([0, 0, 1])
		self.z = randc(1)[0]

class SingleModeCase(TestCase):
	def setUp(self):
		self.basis = array([FockExpansion(e) for e in eye(20)])
		self.z = randc(1)[0]
		self.samples = array([DisplacedState(vac, z, w)
			for z, w in [(0,0), (0,1.8), (0,-3)] + zip(randn(3), randc(3))])
		for ket in self.samples:
			ket.scale(1/sqrt(ket*ket))
					
class ExpansionTest(SingleModeCase):
	def runTest(self):
		for ket in self.samples:
			a = ket.a
			cs = self.basis * ket
			for n in xrange(len(self.basis)):
				assert closenough(cs[n], exp(-0.5*abs(a)**2) * a**n / sqrt(factorial(n)) )
								
class NumberTest(NumberCase):
	def runTest(self):
		assert closenough(2, self.two*self.two.lowered().raised())
		
class CoherentStateBracketTest(SingleModeCase):
	"Verify the inner products of coherent states"
	def runTest(self):
		a = array([q.a for q in self.samples], ndmin=2)
		self.expected = exp(-0.5*abs(a.T)**2 - 0.5*abs(a)**2 + a.conjugate().T*a)
		self.computed = array(outer(self.samples, self.samples), dtype=complex)
		assert closenough(self.expected, self.computed)
				
#class ApproximateDerivativeTest(SingleModeCase):
#	def runTest(self):
#		basis = self.basis
#		hs = self.z*1e-4*eye(2)
#		for ket in self.samples.kets():
#			for h in hs:
#				assert allclose(basis*(ket+h) - basis*(ket-h), dot(basis*ket.D(), 2*h))
#
#class CoherentStateScalingTest(SingleModeCase):
#	"Verify that norms of coherent states are linear in scalar multiplication"
#	def runTest(self):
#		for ket in self.samples.kets():
#			assert allclose(self.basis*(self.z*ket), self.z*(self.basis*ket))



if __name__ == "__main__":
            run_tests()