"""Show the left singular kets of the Jacobian of a random ensemble,
and their coefficients in the projection of the quartic oscillator
Hamiltonian.
"""

from random import uniform, vonmisesvariate
from states import *
import pylab
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import blended_transform_factory
from plot import plotens
import math
from cmath import rect
from utils import randc

R = 8		# ensemble size
ns = 20		# Fock basis size
gs = 10	# size of Q function grid

ham = lop.conjugate()*lop.conjugate()*lop*lop

ext = 4		# size of Q function plots
titlesty = {'va':'top', 'ha':'center', 'fontproperties':FontProperties(size=16)}
lsty = {'va':'center', 'ha':'center', 'fontproperties':FontProperties(size=16)}
csty = {'fontsize':9, 'use_clabeltext':True, 'fmt':'%.2f'}

def randlog(eps):
	"""Return a random complex number, clustered near 0.
	
	The value returned has modulus in [eps, 1], with uniformly distributed logarithm.  Its argument is uniformly distributed.
	"""
	return rect(math.exp(uniform(math.log(eps), 0)), vonmisesvariate(0, 0))

xs = numpy.linspace(2-ext,2+ext,gs)
ys = numpy.linspace(-ext,ext,gs)

def qplot(q):
	qpts = [Bra(coherent(x+y*1j)) for x in xs for y in ys]
	A = Matrix(gs, gs, [abs(a*q)**2 for a in qpts])
	conts = pylab.contour(xs, ys, A, colors='brown', alpha=0.5, linewidths=2)
	pylab.clabel(conts, **csty)
	pylab.axis([2-ext,2+ext,-ext,ext])
	pylab.axis('off')

pylab.figure(1, figsize=(25,12))
pylab.gcf().text(0.5,0.95, "Jacobian of %d random coherent states" % R, **titlesty)
# pylab.gcf().text(0.05, 0.5, r"$\sigma_i$", **lsty)
#T = blended_transform_factory(pylab.gcf().transFigure, pylab.subplot(2,2*R,1).transAxes)
# pylab.text(0.05, 0.5, r"$Q(U_i)$", transform=T, **lsty)
#T = blended_transform_factory(pylab.gcf().transFigure, pylab.subplot(2,2*R,2*R+1).transAxes)
# pylab.text(0.05, 0.5, r"$V_i$", transform=T, **lsty)

pylab.subplot(2,R+1,1)
pylab.xlabel('$|\dot z|$')
pylab.ylabel('residual')

vac = Ket(number(0))
basis = col(Bra(number(n)) for n in xrange(ns))

# choose a random ensemble
esbl = [coherent(2+randc()) for i in xrange(R)]
q = Ket(Sum(*esbl))
q /= norm(q)

# find the SVD of its Jacobian
U, s, V = numpy.linalg.svd(basis*q.D())
V = V.conjugate().T		# Numpy quirk

# Build up zdot one rsv at a time
rhs = (-1j)*basis*ham*q
zdot = numpy.zeros(2*R, dtype=complex)

coefm = abs( col(U[:,2*R-1]).conjugate()*rhs/s[2*R-1] )

for i in xrange(len(s)):
	coef = col(U[:,i]).conjugate()*rhs/s[i]
	zdot += coef*V[:,i]
	
	pylab.figure(1)
	
	# sw
	pylab.subplot(3,4*R+4,4*R+2*i+9)
	pylab.axis('off')
	pylab.ylim(0,s[0])
	pylab.text(0.5, s[i], "%.4f" % s[i], va='center', ha='center')

	# added term
	pylab.subplot(3,4*R+4,4*R+2*i+10)
	pylab.axis('off')
	pylab.ylim(0,coefm)
	pylab.text(0.5, abs(coef), "%.4f" % abs(coef), color='r', va='center', ha='center')

	# cumulative solution
	pylab.subplot(3,2*R+2,4*R+i+7)
	plotens(q)
	qplot(q.D()*col(zdot))
	pylab.axis('off')
	residual = norm((-1j)*ham*q - q.D()*col(zdot))
	pylab.text(1, -4, '%f' % residual, ha='center')
	
	# ensemble
	pylab.subplot(3,2*R+2,i+3)
	plotens(q)
		
	# Q function of left sv
	qplot(basis.conjugate()*col(U[:,i]))

	# L curve
	pylab.subplot(2,R+1,1)
	pylab.loglog(norm(col(zdot)), residual, '.k')

# Q function 
pylab.subplot(3,2*R+2,4*R+5)
plotens(q)
qplot(q)

# Full hamiltonian
pylab.subplot(3,2*R+2,4*R+6)
plotens(q)
qplot((-1j)*ham*q)

pylab.show()