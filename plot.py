# See README.md
#
# In principle, this should be a front end for sciplot.  That involves too much red tape for the moment, so I'm using Metapost.
#
# The Metapost variable u is the scale for coherent amplitudes, and v is the scale for weight circles.


from tempfile import mkdtemp
from os import chdir
from os.path import dirname, join
from subprocess import check_call
from cmath import pi, phase, exp
import numpy as np


def plot_cpt(wgt, amp):
	"Plot a marker for a coherent state with the given coherent amplitude and logarithmic weight-phase."
	
	mpfd.write("mark(%f, %f, %f, %f);\n" %
		(amp.real, amp.imag, abs(exp(wgt)), 180/pi*phase(exp(wgt))))


def plotens(z):
	chdir(mkdtemp())
	global mpfd
	mpfd = open("a.mp", "w")
	plot_begin();
	mpfd.write("u := 1in;\n\nbeginfig(1)\n")
	for r in z:
		plot_cpt(r[0], r[1])
	mpfd.write("draw fullcircle scaled 2u withcolor junk;\n")
	mpfd.write("endfig;\nend\n")
	mpfd.close()
	check_call(["mpost", "a.mp"])
	check_call(["xetex", "mpsproof", "a.1"])
	check_call(["open", "mpsproof.pdf"])


def plot_scale(z):
	"""generate Metapost code to set the scales u and v.  u, the coherent amplitude scale, is set so the markers fit in a 2in square.  v, the weight scale, is set so the maximum radius is 1/3 the median distance to the nearest mark."""
	pass


def plot_begin():
	"""Generate Metapost definitions"""

	mpfd.write("""path marker;
marker = subpath ((3.2/7, 5.37/7)*(length fullcircle)) of fullcircle
	-- (0,0)
	-- subpath ((5.63/7, 7.8/7)*(length fullcircle)) of fullcircle
	-- point (7.8/7*length fullcircle) of fullcircle
	-- 1.1*(point (9/7 length fullcircle) of fullcircle)
	--point (10.2/7*length fullcircle) of fullcircle;

color junk;
junk = 0.7 white;

def mark(expr ax, ay, w, f) =
	draw marker scaled (w*1cm) rotated f shifted (u*(ax, ay));
enddef;\n\n""")

if __name__=="__main__":
	z = np.array([[0,1.8],[2,2+0.2j],[0+1j,2.2]])
	plotens(z)
