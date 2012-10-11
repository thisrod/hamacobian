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
		self.samples = array([DisplacedState(vac, w)
			for w in [0, 1.8, -3] + list(randc(3))])
		for ket in self.samples:
			ket.scale(1/sqrt(ket*ket))
					

class DiracProductAssociates(SingleModeCase):
	def runTest(self):
		k = Ket(Sum(self.samples))
		b = array([Bra(s) for s in self.basis])[:,newaxis]
		z = self.z
		assert closenough((z*b)*k, z*(b*k))
		assert closenough((b*z)*k, b*(z*k))
		assert closenough((b*k)*z, b*(k*z))
	
class ScalarProductCommutes(SingleModeCase):
	def runTest(self):
		k = Ket(Sum(self.samples))
		b = array([Bra(s) for s in self.basis])[:,newaxis]
		z = self.z
		assert closenough(z*b*k, b*z*k)
		assert closenough(z*b*k, b*k*z)

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

class PrmsRoundTrip(SingleModeCase):
	def runTest(self):
		self.s = Sum(self.samples)
		self.zs = self.s.prms()
		self.t = self.s.smrp(self.zs)
		assert abs((self.s-self.t)*(self.s-self.t)) < 1e-8

class UpDown(SingleModeCase):
	"raising a ket and lowering a bra give the same product"
	def runTest(self):
		r = array([s.raised() for s in self.samples])
		l = array([s.lowered() for s in self.samples])
		sm = Sum(self.samples)
		assert closenough(l*self.samples, self.samples*r)
		assert closenough(l*sm, self.samples*sm.raised())
		assert closenough(sm.lowered()*self.samples, sm*r)
		assert closenough(sm.lowered()*sm, sm*sm.raised())
		
		
				
class ApproximateDerivativeTest(SingleModeCase):
	def runTest(self):
		basis = array([Bra(s) for s in self.basis])[:,newaxis]
		q = Ket(Sum(self.samples))
		zs = q.prms()
		hs = self.z*1e-4*eye(len(zs))
		self.expected = empty((basis.size, zs.size), dtype=complex)
		for n in xrange(len(zs)):
			self.expected[:,n:n+1] = basis*(q.smrp(zs+hs[n,:]) - q.smrp(zs-hs[n,:]))
		self.computed = 2*hs[0,0] * dot(basis, q.D())
		assert closenough(self.expected, self.computed)

#class CoherentStateScalingTest(SingleModeCase):
#	"Verify that norms of coherent states are linear in scalar multiplication"
#	def runTest(self):
#		for ket in self.samples.kets():
#			assert allclose(self.basis*(self.z*ket), self.z*(self.basis*ket))



if __name__ == "__main__":
            run_tests()