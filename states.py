from brackets import combinations
from numpy import array, empty, log, sqrt, sum
import numbers
from copy import copy, deepcopy


##################################################
#
#	Product dispatch system
#
##################################################

def mul(self, other):
	if isinstance(other, numbers.Complex):
		return self.scaled(other)
	elif (type(self), type(other)) in products:
		return products[type(self), type(other)](self, other)
	else:
		return NotImplemented
		
def rmul(self, other):
	if isinstance(other, numbers.Complex):
		return self.scaled(other)
	elif (type(self), type(other)) in products:
		return products[type(self), type(other)](self, other).conj().T
	else:
		return NotImplemented
	
	
##################################################
#
#	Row and sum classes
#
##################################################

class KetRow(object):
	
	def similar(self, other):
		return type(self) is type(other) and \
			len(self) == len(other) and \
			wid(self) == wid(other)
			
	def norm(self):
		return sqrt(sum(self*self).real)
		
	def sum(self):
		return KetSum(self)
		
	__mul__, __rmul__= mul, rmul
	

def wid(s):
	return s.__wid__()
	
	
class KetSum(object):

	def __init__(self, row):
		self.components = row
		
	def expand(self, basis):
		"basis must be a row of orthonormal kets, but this could be relaxed"
		return type(basis)(basis*self)
		
	def _sum_mul(self, other):
		return sum(self.components * other.components)
		
	def _row_mul(self, other):
		return sum(self.components*other, axis=1)
		
	def scaled(self, z):
		return KetSum(self.components.scaled(z))
	
	__mul__, __rmul__= mul, rmul
	
	
##################################################
#
#	Specific types of sum
#
##################################################

class LccState(KetRow):
	"A linear combination of coherent states."
	
	# f is row of log weights, and a is a matrix whose cobrackets.lumns are vector ampltiudes.  these shapes fit a row of kets.
	# Use copy construction if efficiency matters

	def __init__(self, *args):
		"LccState(f1, a1, ..., fn, an) where f is a logarithmic weight, and a the corresponding coherent amplitude(s)."
		if isinstance(args[1], numbers.Complex):
			ain = [[x] for x in args[1::2]]
		else:
			ain = args[1::2]
		n = len(args)/2
		m = len(ain[0])+1
		self.setz(empty((m,n), dtype=complex))
		self.setf(args[0::2])
		self.seta(ain)
		
	def __len__(self):
		return self.z.shape[1]
		
	def __wid__(self):
		return self.z.shape[0]
		
	def setz(self, z):
		self.z = z
		self.f = self.z[0:1,:]
		self.a = self.z[1:,:]
		
	def setf(self, f):
		self.f[:,:] = f
		
	def seta(self, a):
		# This takes amplitude vectors as rows, as does __init__
		self.a[:,:] = array(a).T
		
	def __deepcopy__(self, d):
		result = copy(self)
		result.setz(self.z.copy())
		return result
		
	def normalised(self):
		result = deepcopy(self)
		result.f -= log(self.norm())
		return result
		
	def D(self):
		result = DLccState()
		result.be(self)
		return result
		
	def __add__(self, dz):
		result = deepcopy(self)
		result.z += dz
		return result
		

class DLccState(KetRow):
	"the total derivative of an LccState wrt z.  forms products with states and number state vectors as 2D arrays."

	def be(self, state):
		self.state = state
		
	def __len__(self):
		return len(self.state)*wid(self.state)
		
	def __wid__(self):
		# No adjustable parameters
		return 0


class NState(KetRow):
	"a state expanded over Fock states."
	
	def __init__(self, *cs):
		"cs are the coefficients, starting with |0>"
		self.cs = array(cs, dtype=complex, ndmin=2)
		
	def __len__(self):
		return self.cs.shape[1]
		
	def __wid__(self):
		return 1
		
		
##################################################
#
#	Dispatch table
#
##################################################

products = {}

products[KetSum, KetSum] = KetSum._sum_mul
	
for T in KetRow.__subclasses__():
	products[KetSum, T] = KetSum._row_mul
	
for Ts, op in combinations(NState, LccState, DLccState):
	products[Ts] = op
