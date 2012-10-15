# Investigate how well derivatives approximate linear differences.

from random import gauss
from sys import argv
from states import *
import pylab

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = int(argv[1])		# number of states in ensemble
dst = [abs(exp(x)) for x in pylab.linspace(log(1e-4), log(5), 50)]

# choose a random ensemble
v = FockExpansion(1)
q = Ket(Sum(*(DisplacedState(randc(), v) for i in xrange(R))))

for p in xrange(7):
	# choose a random direction
	drn = col(randc() for i in q.prms().indices())
	drn /= norm(drn)
	
	# plot how the approximation fails with distance
	rslt = {}
	for h in dst:
		shft = q.smrp(q.prms()+h*drn) - q
		lin = (q.D()*h*drn).elts[0]
		rslt[h] = norm(shft - lin)/norm(lin)
		
	x = rslt.keys()
	x.sort()
	pylab.loglog(x, [rslt[i] for i in x])
	pylab.hold(True)
	
pylab.title('%d coherent states linearly approximated in random directions' % R)
pylab.xlabel('norm of parameter shift')
pylab.ylabel('relative error in linear approximation')
	
pylab.show()