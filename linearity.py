# Investigate how well derivatives approximate linear differences.

from random import gauss
from sys import argv
from states import *
import pylab

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = int(argv[1])		# number of states in ensemble
dst = [abs(exp(x)) for x in pylab.linspace(log(1e-5), log(5), 5)]
basis = col(Bra(number(n)) for n in xrange(20))

# choose a random ensemble
v = FockExpansion(1)
q = Ket(Sum(*(DisplacedState(randc(), v) for i in xrange(R))))

m = []	# slopes
rpts = 7

pylab.subplot(211)
pylab.title('%d coherent states linearly approximated in random directions' % R)
pylab.xlabel('norm of parameter shift')
pylab.ylabel('relative error in linear approximation')

for p in xrange(rpts):
	# choose a random direction
	drn = col(randc() for i in q.prms().indices())
	drn /= norm(drn)
	
	# plot how the approximation fails with distance
	sqt = []
	bss = []
	for h in dst:
		shft = q.smrp(q.prms()+h*drn) - q
		lin = q.D()*h*drn
		sqt.append( norm(shft - lin)/norm(lin) )
		bss.append( norm(basis*(shft - lin))/norm(basis*lin) )

	# slope in the quadratic region between 1e-3 and 1e-2
#	m.append( (log(sqt[21]) - log(sqt[10])) / \
#		(log(dst[21]) - log(dst[10])) )

	pylab.subplot(211)
	pylab.loglog(dst, sqt)
	pylab.hold(True)
	pylab.subplot(212)
	pylab.loglog(dst, bss)
	pylab.hold(True)
	
#pylab.text(0.001, 0.01, "slope %.3f pm %.3f" % 
#	(sum(m).real/len(m), sqrt(sum(x**2 for x in m)/rpts - sum(m)**2/rpts**2).real))

pylab.show()