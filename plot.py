# See README.md

from tempfile import mkdtemp
from os.path import join
from subprocess import check_call

mpscript = """input plot;

beginfig(1)
	draw marker scaled 1in;
endfig;
end
"""

def plotens():
	amp = join(mkdtemp(), "a.mp")
	mpfd = open(amp, "w")
	print mpfd
	mpfd.write(mpscript)
	mpfd.close()
	check_call(["mpost", amp])
	check_call(["xetex", "mpsproof", "a.1"])
	check_call(["open", "mpsproof.pdf"])