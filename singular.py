# Plot the singular vectors of the Jacobian of random ensembles of coherent states

from random import gauss
from states import *
import pylab
import scipy.interpolate
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import blended_transform_factory

def randc():
	return gauss(0,1) + gauss(0,1)*1j

R = 3		# ensemble size
p = 20		# maximum particle number
gbase = 200 + 2*R*10 + 1
gs = 10	# size of Q function grid

ext = 4		# size of Q function plots
titlesty = {'va':'top', 'ha':'center', 'fontproperties':FontProperties(size=16)}
lsty = {'va':'center', 'ha':'center', 'fontproperties':FontProperties(size=16)}
csty = {'fontsize':9, 'use_clabeltext':True, 'fmt':'%.2f'}

pylab.figure(figsize=(25,12))
pylab.gcf().text(0.5,0.95, "Jacobian of %d random coherent states" % R, **titlesty)
pylab.gcf().text(0.05, 0.5, r"$\sigma_i$", **lsty)
T = blended_transform_factory(pylab.gcf().transFigure, pylab.subplot(2,2*R,1).transAxes)
pylab.text(0.05, 0.5, r"$Q(U_i)$", transform=T, **lsty)
T = blended_transform_factory(pylab.gcf().transFigure, pylab.subplot(2,2*R,2*R+1).transAxes)
pylab.text(0.05, 0.5, r"$V_i$", transform=T, **lsty)

vac = Ket(number(0))
basis = col(Bra(number(n)) for n in xrange(p))

# choose a random ensemble
esbl = [coherent(randc()) for i in xrange(R)]
q = Ket(Sum(*esbl))
q /= norm(q)
U, s, V = numpy.linalg.svd(basis*q.D())
V = V.conjugate().T		# Numpy quirk
qas = [st.a for st in esbl]
xs = numpy.linspace(-ext,ext,gs)
ys = numpy.linspace(-ext,ext,gs)

for i in xrange(len(s)):
	# sw
	pylab.subplot(2,2*R,2*R+i+1)
	T = blended_transform_factory(pylab.gca().transAxes, pylab.gcf().transFigure)
	pylab.text(0.5, 0.5, "%.4f" % s[i], transform=T)

	# right sv
	pylab.axis('off')
	pylab.bar(range(2*R), V[:,i].real, 0.35, hold=True, color='k')
	pylab.bar(numpy.arange(2*R)+0.35, V[:,i].imag, 0.35, color='r')
	pylab.ylim(-1,1)
	
	# ensemble
	pylab.subplot(2,2*R,i+1)
	ax = pylab.gca()
	circ = pylab.Circle((0, 0), 1, color='k', alpha=0.1)
	ax.add_patch(circ)
	for j in xrange(len(qas)):
		pylab.text(qas[j].real, qas[j].imag, "%d" % j, va='center', ha='center')
		
	# Q function of left sv
	qpts = [Bra(coherent(x+y*1j)) for x in xs for y in ys]
	A = Matrix(gs, gs, [abs(a*basis.conjugate()*col(U[:,i]))**2 for a in qpts])
	conts = pylab.contour(xs, ys, A, colors='brown', alpha=0.5, linewidths=2)
	pylab.clabel(conts, **csty)
	pylab.axis([-ext,ext,-ext,ext])
	pylab.axis('off')

pylab.show()