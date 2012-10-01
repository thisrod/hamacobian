# Refactored matrices, bras and kets

##################################################
#
#	operator dispatch
#
##################################################

class QO(object):
	pass

##################################################
#
#	abstract quantum states
#
##################################################

class States(QO):
	"a collection of similar quantum states"
	pass

	def bras(self):
		# do I return Kets, or a Row?
		pass

	def kets(self):
		pass

class Bras(QO):
	pass
	
class Kets(QO):
	pass
	
class Sum(States):
	"a collection with one element"
	pass

##################################################
#
#	Hilbert space matrices
#
##################################################
	
class Row(QO):
	pass
	
class Col(QO):
	pass

# how do these combine with ndarray?

##################################################
#
#	representations of quantum states
#
##################################################

class FockExpansions(states):
	pass
	
class DisplacedStates(states):
	pass
