"""Test hypotheses about stiffness of quartic oscillator equations"""

from random import uniform, vonmisesvariate
import math
from states import *
import pylab
import numpy as np
from cmath import rect

def randlog(eps):
	"""Return a random complex number, clustered near 0.
	
	The value returned has modulus in [eps, 1], with uniformly distributed logarithm.  Its argument is uniformly distributed.
	"""
	
	return rect(math.exp(uniform(math.log(eps), 0)), vonmisesvariate(0, 0))

eps = 1e-3	# minimum separation
lbda = 1e-2	# regularisation strength
h = 1e-5		# range to shift z
R = 4			# size of ensemble
p = 20		# basis size
ensamp = 3	# number of ensembles to sample
treg = lbda*np.eye(2*R)	# Tychonov

basis = col(Bra(number(n)) for n in xrange(p))
v = FockExpansion(1)

ham = lop.conjugate()*lop.conjugate()*lop*lop
	
def randd(n):
	"""Return a unit n-vector in a random direction."""
	drn = col(randc() for i in q.prms().indices())
	return drn / norm(drn)

def setlrhs(q):
	global lhs, rhs
	lhs = basis*q.D()
	iHq = (-1j)*(ham*q)
	rhs = basis*iHq
	

def zdot(reg=None):
	"return zdot such that  [|q>zdot; reg] = -iH|q>"
	global lhs, rhs
	if reg is not None:
		lhs = np.vstack((lhs, reg))
		rhs = np.vstack((rhs, np.zeros_like(reg[:,0:1])))
	zd, r, rk, sing = np.linalg.lstsq(lhs, rhs)
	zd = col(zd.flatten())
	return zd
	
def zdotiter(q):
	"""Iterative regularisation"""
	zd = col([0]*2*R)
	for i in xrange(4):
		V = np.array(q.D().conjugate()*q.D())
		iHq = (1j)*(ham*q)
		rhs = np.array(q.D().conjugate()*iHq).flatten() + np.dot(V, zd).flatten()
		zd = np.linalg.solve(V+1j* lbda**2 * np.eye(2*R), rhs)
	return col(zd.flatten())
		

hms = []		# reciprocal harmonic mean separt	
cn = []		# bare condition number of |Dpsi>
tcn = []		# t = tychonof conditioned
scn = []		# s = softened
jacr = []		# condition number of bare z->zdot Jacobian
tjacr = []
ijacr = []		# i = iterative
sjacr = []		# 

for ss in xrange(ensamp):

	# choose a random ensemble about amplitude 2
	qas = [2+randlog(eps) for i in xrange(R)]
	q = Ket(Sum(*[DisplacedState(a, v) for a in qas]))
	q /= norm(q)
	bqd = np.array(basis*q.D())

	cn.append(numpy.linalg.cond(bqd))
	tcn.append(numpy.linalg.cond(np.vstack((bqd, treg))))
	hms.append(sum(1/abs(qas[i]-qas[j]) for i in xrange(R) for j in xrange(i)))
	# actually 1/hms

	# softening.  assume all components are nearly parallel, and constrain them all
	sreg = np.empty((0,2*R), dtype=complex)
	for i in xrange(2*R):
		for j in xrange(i):
			blk = np.zeros((p,2*R), dtype=complex)
			blk[:,i] = bqd[:,i]
			blk[:,j] = -bqd[:,j]
			sreg = np.vstack((sreg,blk))
	sreg *= lbda
	scn.append(numpy.linalg.cond(np.vstack((bqd, sreg))))
	
	setlrhs(q)
	zd = zdot()
	tzd = zdot(treg)
	szd = zdot(treg)
	izd = zdotiter(q)
	zdjac = np.empty((2*R,0))
	tzdjac = np.empty((2*R,0))
	szdjac = np.empty((2*R,0))
	izdjac = np.empty((2*R,0))
	z = q.prms()
	for x in np.eye(2*R):
		setlrhs(q.smrp(z+h*col(x)))
		br = (zdot()-zd)/h
		tr = (zdot(treg)-tzd)/h
		sr = (zdot(sreg)-szd)/h
		ir = (zdotiter(q.smrp(z+h*col(x)))-izd)/h
		
		setlrhs(q.smrp(z+1j*h*col(x)))
		bi = (zdot()-zd)/h
		ti = (zdot(treg)-tzd)/h
		si = (zdot(sreg)-szd)/h
		ii = (zdotiter(q.smrp(z+1j*h*col(x)))-izd)/h
		
		zdjac = np.hstack((zdjac, br, bi))
		tzdjac = np.hstack((tzdjac, tr, ti))
		szdjac = np.hstack((tzdjac, sr, si))
		izdjac = np.hstack((izdjac, ir, ii))
	jacr.append(np.linalg.cond(zdjac))
	tjacr.append(np.linalg.cond(tzdjac))
	sjacr.append(np.linalg.cond(szdjac))
	ijacr.append(np.linalg.cond(izdjac))


pylab.subplot(221)
pylab.ylabel("stiffness ratio")
pylab.loglog(hms, jacr, 'k.')
pylab.loglog(hms, tjacr, 'r.')
pylab.loglog(hms, sjacr, 'g+')
pylab.loglog(hms, ijacr, 'b.')

pylab.subplot(222)
pylab.xlabel("Jacobian condition")
pylab.loglog(cn, jacr, 'k.', hold=True)
pylab.loglog(cn, ijacr, 'b.')

pylab.subplot(223)
pylab.xlabel("r.h.m separation")
pylab.ylabel("Jacobian condition")
pylab.loglog(hms, cn, 'k.', hold=True)
pylab.loglog(hms, tcn, 'r.')
pylab.loglog(hms, scn, 'g+')

pylab.show()