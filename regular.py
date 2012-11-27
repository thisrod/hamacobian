"""Investigate the performance of different regularisation methods for the quartic oscillator"""

from random import gauss
from states import *
import pylab
from plot import plotens

def randc(): return gauss(0,1) + gauss(0,1)*1j

dmax = 1e-5		# range to shift z
R = 4			# size of ensemble
gs = 10		# points to plot
rpts = 10		# directions to try
p = 20		# basis size
sig = 0.3		# standard deviation of initial ensemble

ham = lop.conjugate()*lop.conjugate()*lop*lop

def zdot(ql, qr):
	"return zdot such that |ql>zdot = -iH|qr>"
	lhs = basis*ql.D()
	iHq = (-1j)*(ham*qr)
	rhs = basis*iHq
	zd, r, rk, sing = numpy.linalg.lstsq(lhs, rhs)
	zd = col(zd.flatten())
	return zd
	
# choose a random ensemble about amplitude 2
basis = col(Bra(number(n)) for n in xrange(p))
v = FockExpansion(1)
q = Ket(Sum(*(DisplacedState(2+sig*randc(), v) for i in xrange(R))))
q /= norm(q)
x = [float(x) for x in pylab.linspace(0,dmax, gs)]
x = x[1:]		# avoid plotting log(0)

zd = zdot(q,q)
foo, sing, bar = numpy.linalg.svd(basis*q.D())
	
pylab.subplot(321)
pylab.title("changing projection")
pylab.subplot(323)
pylab.title("changing state")
pylab.ylabel("$|\dot{z} - \dot{z}_0|$", size=16)
pylab.subplot(325)
pylab.title("changing both")
pylab.xlabel("norm of change in z")
pylab.subplot(324)
pylab.title("2 norm of Jacobian change")

pylab.subplot(343)
pylab.title("ensemble relative to unit circle")
plotens(q)

pylab.subplot(344)
pylab.title("singular values of Jacobian")
pylab.semilogy(sing, 'o', mfc='none')
pylab.xlim([-0.5, sing.size-0.5])

pylab.gcf().text(0.55,0.2, "relative residual: %.2f" % (norm(q.D()*zd - (-1j)*ham*q)/norm(ham*q)))
pylab.gcf().text(0.55,0.3, "norm zdot: %.1f" % norm(zd))

pylab.gcf().text(0.5,0.95, "Change of zdot for quartic oscillator, as z changes in random directions", ha='center')

for k in xrange(rpts):
	left = []
	right = []
	both = []
	cnum = []
	
	# choose a random direction
	drn = col(randc() for i in q.prms().indices())
	drn /= norm(drn)
	
	for eps in x:
		q1 = q.smrp(q.prms()+eps*drn)
		left.append(norm(zd-zdot(q1,q)))
		right.append(norm(zd-zdot(q,q1)))
		both.append(norm(zd-zdot(q1,q1)))
		cnum.append(numpy.linalg.norm(basis*(q1.D()-q.D()), 2))
	
	pylab.subplot(321)
	pylab.plot(x, left, hold=True)
	pylab.subplot(323)
	pylab.plot(x, right, hold=True)
	pylab.subplot(325)
	pylab.plot(x, both, hold=True)
	pylab.subplot(324)
	pylab.plot(x, cnum, hold=True)
	
pylab.show()


# print "%.3f, %3.0f, %3.0f, %3.0f" % (rerr, norm(zdot), norm(iHq), 
# 	sum(norm(Ket(t)) for t in iHq.s.terms()))
