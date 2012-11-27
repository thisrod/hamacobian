"""Test hypotheses about stiffness of quartic oscillator equations"""

from random import gauss
from states import *
from utils import *
import pylab
import numpy as np

def randc(): return gauss(0,1) + gauss(0,1)*1j

h = 1e-5		# range to shift z
R = 4			# size of ensemble
p = 20		# basis size
sig = 0.3		# standard deviation of initial ensemble
ensamp = 30	# number of ensembles to sample

basis = col(Bra(number(n)) for n in xrange(p))
v = FockExpansion(1)

ham = lop.conjugate()*lop.conjugate()*lop*lop
	
def randd(n):
	"""Return a unit n-vector in a random direction."""
	drn = col(randc() for i in q.prms().indices())
	return drn / norm(drn)

def zdot(ql, qr):
	"return zdot such that |ql>zdot = -iH|qr>"
	lhs = basis*ql.D()
	iHq = (-1j)*(ham*qr)
	rhs = basis*iHq
	zd, r, rk, sing = np.linalg.lstsq(lhs, rhs)
	zd = col(zd.flatten())
	return zd

cn = []
hms = []
jacr = []

for ss in xrange(ensamp):

	# choose a random ensemble about amplitude 2
	qas = [2+sig*randc() for i in xrange(R)]
	q = Ket(Sum(*[DisplacedState(a, v) for a in qas]))
	q /= norm(q)

	zd = zdot(q,q)
	cn.append(numpy.linalg.cond(basis*q.D()))
	hms.append(sum(1/abs(qas[i]-qas[j]) for i in xrange(R) for j in xrange(i)))
	# actually 1/hms
	
	zdjac = np.empty((2*R,0))
	z = q.prms()
	zd = zdot(q,q)
	for x in np.eye(2*R):
		qre = q.smrp(z+h*col(x))
		qim = q.smrp(z+1j*h*col(x))
		zdjac = np.hstack((zdjac, (zdot(qre,qre)-zd)/h, (zdot(qim,qim)-zd)/h))
	jacr.append(np.linalg.cond(zdjac))


pylab.subplot(221)
pylab.ylabel("1/h.m. separation")
pylab.semilogx(cn, hms, '.')

pylab.subplot(222)
pylab.xlabel("stiffness ratio")
pylab.semilogx(jacr, hms, '.')

pylab.subplot(223)
pylab.xlabel("Jacobian condition")
pylab.ylabel("stiffness ratio")
pylab.loglog(cn, jacr, '.')

pylab.show()