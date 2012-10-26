# examine Tychonov solutions in Fock space

from random import gauss
from states import *
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import blended_transform_factory
import numpy

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = 3		# number of states in ensemble
p = 20			# maximum particle number
x = [abs(exp(x)) for x in pylab.linspace(log(1e-4), log(30), 50)]
rx = [1./y for y in x]

# choose a random ensemble
vac = Ket(number(0))
basis = col(Bra(number(n)) for n in xrange(p))

marked = False
for i in xrange(9):
	q = Ket(Sum(*(coherent(randc()) for i in xrange(R))))
	q /= norm(q)
	nprms = q.prms().ht
	
	# expand over a Fock basis, and find the best approximation to the vacuum state
	
	U, s, V = numpy.linalg.svd(basis*q.D())
	vmin = row(V[-1,:])	# take advantage of numpy quirk
	shft = []		# norm of parameter shift
	smin = []		# most singular component of shift
	rlr = []		# residual reported by numpy
	clr = []		# residual computed
	nlr = []		# nonlinear residual
	for eps in x:
		lhs = numpy.vstack((basis*q.D(), eps*numpy.eye(nprms)))
		rhs = numpy.vstack((basis*(vac-q), numpy.zeros((nprms,1))))
		h, r, rk, sing = numpy.linalg.lstsq(lhs, rhs)
		h = col(h.flatten())
		r -= norm(eps*h)**2		# adjust for Tychonov
		V1 = q.smrp(q.prms() + h)
		V2 = q + q.D()*h
		shft.append(norm(h))
		smin.append(abs(vmin*h))
		rlr.append(sqrt(r))
		clr.append(norm(vac-V2))
		nlr.append(norm(vac-V1))
	
	pylab.subplot(330+i)
	
	trust = min(x[i] for i in xrange(len(x)) if nlr[i] < nlr[-1])
#	pylab.axvline(trust, linewidth=2, color='k', ymax=0.9)
#	if not marked and trust > x[0]:
#		T = blended_transform_factory(pylab.gca().transData, pylab.gca().transAxes)
#		pylab.text(trust, 0.7, "  trust region", transform=T, ha='left')
#		marked = True
	pylab.semilogx(rx, shft, color='blue')
	pylab.semilogx(rx, smin, '--', color='blue')
	pylab.semilogx(rx, nlr, color='black')
	pylab.semilogx(rx, clr, color='brown')
	if max(nlr) > 15: pylab.ylim([0,15])
	pylab.text(1, 1, r"$\sigma > %.4f$" % s[-1], transform = pylab.gca().transAxes, va='top', ha='right')

pylab.gcf().text(0.5, 1, r"Fitting %d random coherent states to $|0\rangle$" % R, va='top', ha='center', fontproperties=FontProperties(size=16))
pylab.gcf().text(0.5, 0.95, "blue = parameter shift, dashed = $\sigma$ cpt, brown = linear error, black = nonlinear error", va='top', ha='center')
pylab.subplot(338)
pylab.xlabel(r"trust length $1/\epsilon$")
pylab.subplot(334)
pylab.ylabel("distance")
pylab.show()
