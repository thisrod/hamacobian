"""Instrumented integrator for the quartic oscillator"""

from states import *
import pylab
import scipy.integrate
from random import gauss
from numpy import array
from plot import plotens

def randc(): return gauss(0,1) + gauss(0,1)*1j

tmax = 1e-3
R = 3
ham = lop.conjugate()*lop.conjugate()*lop*lop
p = 20		# Fock basis size

basis = col(Bra(number(n)) for n in xrange(p))
def zdot(t, z):
	"return zdot such that |ql>zdot = -iH|qr>"
	q = q0.smrp(col(z))
	lhs = basis*q.D()
	iHq = (-1j)*ham*q
	rhs = basis*iHq
	zd, r, rk, sing = numpy.linalg.lstsq(lhs, rhs)
	zd = zd.flatten()
	print zd
	return zd

# set up initial state
v = FockExpansion(1)
q0 = Ket(Sum(*(DisplacedState(a, v) for a in [ 2.46261965+0.70381217j,  2.48197955-0.16273579j,
        2.39546681+0.22578992j])))
q0 /= norm(q0)

pylab.subplot(211)
plotens(q0)

igtr = scipy.integrate.complex_ode(zdot)
igtr.set_initial_value(array(q0.prms()).flatten())
pylab.subplot(212)
plotens(q0.smrp(col(igtr.integrate(tmax))))

pylab.show()
