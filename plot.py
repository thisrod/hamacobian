"""Pylab extensions for ensembles of coherent states"""

import pylab
import numpy as np

def plotens(q):
	"""Plot the coherent amplitudes of q, which can be a bra, ket, or sum state."""
	
	a = np.array(q.prms()).flatten()[1::2]
	ax = pylab.gca()
	circ = pylab.Circle((0, 0), 1, color='k', alpha=0.1)
	ax.add_patch(circ)
	ax.axis('equal')
	for j in xrange(a.size):
		pylab.text(a[j].real, a[j].imag, "%d" % j, va='center', ha='center')
	a = np.concatenate((a, [0]))
	pylab.axis([min(a.real)-1, max(a.real)+1, min(a.imag)-1, max(a.imag)+1])
	pylab.axis('off')
