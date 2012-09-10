# See README.md
#
# In principle, this should be a front end for sciplot.  That involves too much red tape for the moment, so I'm using Metapost.


from tempfile import mkdtemp
from os.path import join
from subprocess import check_call
from cmath import pi, phase, exp

mpprefix = """input plot;

beginfig(1)
"""

mpsuffix = """endfig;
end
"""

def plot_cpt(wgt, amp):
	"Plot a marker for a coherent state with the given coherent amplitude and logarithmic weight-phase."
	
	mpfd.write("mark(%f, %f, %f, %f);\n" %
		(amp.real, amp.imag, abs(exp(wgt)), 180/pi*phase(exp(wgt))))

def plotens(z):
	global mpfd
	amp = join(mkdtemp(), "a.mp")
	mpfd = open(amp, "w")
	print mpfd
	mpfd.write(mpprefix)
	for r in z:
		plot_cpt(r[0], r[1])
	mpfd.write(mpsuffix)
	mpfd.close()
	check_call(["mpost", amp])
	check_call(["xetex", "mpsproof", "a.1"])
	check_call(["open", "mpsproof.pdf"])