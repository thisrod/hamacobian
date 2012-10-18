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

m = []	# slopes
rpts = 7

pylab.figure("Curvature")
pylab.title('%d coherent states linearly approximated in random directions' % R)
pylab.xlabel('norm of parameter shift')
pylab.ylabel('relative error in linear approximation')

for p in xrange(rpts):
	# choose a random direction
	drn = col(randc() for i in q.prms().indices())
	drn /= norm(drn)
	
	# plot how the approximation fails with distance
	rslt = {}
	for h in dst:
		shft = q.smrp(q.prms()+h*drn) - q
		lin = q.D()*h*drn
		rslt[h] = norm(shft - lin)/norm(lin)

	# slope in the quadratic region between 1e-3 and 1e-2
	m.append( (log(rslt[dst[21]]) - log(rslt[dst[10]])) / \
		(log(dst[21]) - log(dst[10])) )
		
	x = rslt.keys()
	x.sort()
	pylab.loglog(x, [rslt[i] for i in x])
	pylab.hold(True)
	
pylab.text(0.001, 0.01, "slope %.3f pm %.3f" % 
	(sum(m).real/len(m), sqrt(sum(x**2 for x in m)/rpts - sum(m)**2/rpts**2).real))

pylab.figure("Expansions")

dst = [abs(exp(x)) for x in pylab.linspace(log(1e-10), log(1e-2), 50)]
	
p = 7		# maximum particle number
r1 = [ [] for i in xrange(p) ]
r2 = [ [] for i in xrange(p) ]
r3 = [ [] for i in xrange(p) ]

for h in dst:
	shft = q.smrp(q.prms()+h*drn) - q
	lin = q.D()*h*drn
	for n in xrange(p):
		r1[n].append(abs(Bra(number(n))*shft))
		r2[n].append(abs(Bra(number(n))*lin))
		r3[n].append(abs(Bra(number(n))*(shft - lin)))

pylab.subplot(311)		
pylab.hold(True)
for i in xrange(p):
	pylab.loglog(dst, r1[i])
pylab.title('finite difference')

pylab.subplot(312)		
pylab.hold(True)
for i in xrange(p):
	pylab.loglog(dst, r2[i])
pylab.ylabel(r'weight of $|n\rangle$')
pylab.title('scaled derivative')

pylab.subplot(313)		
pylab.hold(True)
for i in xrange(p):
	pylab.loglog(dst, r3[i])
pylab.title('discrepency')
	
pylab.xlabel('norm of parameter shift')
pylab.show()