# compare least squares solutions in fock space to those in coherent state normal equations

from random import gauss
from states import *
import pylab
import numpy

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = 3		# number of states in ensemble
N = 20			# maximum particle number

nop = lop.conjugate() * lop
X1 = 0.5*(lop.conjugate() + lop)
X2 = 0.5*(lop.conjugate() - lop)


# choose a random ensemble
vac = Ket(number(0))
q = Ket(Sum(*(coherent(randc()) for i in xrange(R))))
q /= norm(q)

# expand over a Fock basis, and find the best approximation to the vacuum state
basis = col(Bra(number(n)) for n in xrange(N))
dz, r, rk, sing = numpy.linalg.lstsq(basis*q.D(), basis*(vac-q))
dz = col(dz.flatten())
V1 = q.smrp(q.prms() + dz)
V2 = q + (q.D()*dz).elts[0]