# Investigate how well derivatives approximate linear differences.

from random import gauss
from sys import argv
from states import *
import pylab

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = int(argv[1])		# number of states in ensemble
dst = [abs(exp(x)) for x in pylab.linspace(log(1e-8), log(8), 50)]
basis = col(Bra(number(n)) for n in xrange(20))

# choose a random ensemble
v = FockExpansion(1)
q = Ket(Sum(*(DisplacedState(randc(), v) for i in xrange(R))))

m = []	# slopes
i2 = max(i for i in xrange(len(dst)) if dst[i] < 1e-2)		# quadratic region of graph
i1 = min(i for i in xrange(len(dst)) if dst[i] > 1e-6)
rpts = 7

pylab.title('%d coherent states linearly approximated in random directions' % R)
pylab.xlabel('norm of parameter change')
pylab.ylabel('relative error in linear approximation')

for p in xrange(rpts):
	# choose a random direction
	drn = col(randc() for i in q.prms().indices())
	drn /= norm(drn)
	
	# plot how the approximation fails with distance
	bss = []
	for h in dst:
		shft = q.smrp(q.prms()+h*drn) - q
		lin = q.D()*h*drn
		# This small difference more stable if we expand over an orthonormal basis
		bss.append( norm(basis*(shft - lin))/norm(basis*lin) )

	# slope in the quadratic region between 1e-3 and 1e-2
	m.append( (log(bss[i2]) - log(bss[i1])) / \
		(log(dst[i2]) - log(dst[i1])) )

	pylab.loglog(dst, bss)
	pylab.hold(True)
	
pylab.text(0.5, 0.7, r"slope $%f \pm %f$" % 
	(sum(m).real/len(m), sqrt(sum(x**2 for x in m)/rpts - sum(m)**2/rpts**2).real),
	transform = pylab.gca().transAxes)

pylab.show()