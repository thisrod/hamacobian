Design of the states and patterns modules
======

This file describes the libraries that implement simulations by variation of paths.  The theory behind the method, and its numerical stability, are described elsewhere.

The variation of paths method uses sums of coherent states, and the derivatives of those sums with respect to the coefficients and coherent amplitudes.  It requires the calculation of brackets of these, and moments of the Hamiltonian between them.  This calculation is repeated at every timestep of a simulation.


Matrix and scalar patterns
----------------

Patterns are a kind of method object.  The simplest type is the scalar pattern: these take an (m,n) vector of parameters as input, and return an (m) array of results as output.  The pattern represents some function f, and each element of the output is f(x[i,:]) for a row of input.  Some patterns take two parameter vectors, and return an (m1,m2) array whose elements are f(x1[i,:],x2[j,:]).

The most straightforward scalar patterns are polynomials of the parameters, and the exponential that emerges from inner products of coherent states.  There are also scalar patterns that form bras, kets and operators; these are discussed below.

Matrix patterns are formed from scalar patterns, whose output they interleave to form block arrays.  For example, a row pattern can be formed from three scalar patterns implementing functions f, g, and h.  It takes an (m,n) vector as input, and returns a (3*m) array (f(x[1,:]), g(x[1,:]), h(x[1,:]), …).  Optionally, it could have a block pattern p, in which case it would return (p(x[1,:])*f(x[1,:]), p(x[1,:])*g(x[1,:]), p(x[1,:])*h(x[1,:]), …).

The shape of a pattern is a tuple of the lengths of the parameter vectors it accepts.  Two patterns are similar if they have the same shape.

Multiplying two patterns with |*| identifies their variables with each other.  The |mvmul| method forms a product where the variables are independent.

Column patterns are very similar, except that the scalar patterns are applied to conjugated parameters.  Outer product patterns take a matrix of scalar patterns, each of which takes two sets of parameters; they return a matrix where the first argument varies down columns, and the second across rows.  They also take a block pattern.  Sum patterns form a row or matrix, and sum the elements rowwise.

Bras and kets are implemented by a type of scalar pattern called a state.  These act as bras when formed into columns, and kets when formed into rows.  When acting as a bra, a state expects to receive conjugated parameters, like any column.  The parameters of the state are scalar patterns for now; when we go multimode, coherent amplitudes might become structures.

Matrix patterns can be multiplied by scalar patterns or matrix patterns, according to the usual rules.  States are treated as bras when multiplying rows, and kets when multiplying columns.

Numbers and quantum operators are treated as constant scalar patterns.

Derivatives still need to be sorted out.  There are two issues: arrays of parameters, and Wertinger derivatives.

To evaluate moments, we need to add up the leading nonzero terms of Taylor series.  That requires a method |isZero| to tell when patterns are zero.  It returns |True| if the pattern evaluates to zero for all choices of input parameters.

Patterns should be optimised for fast substitution.  Perhaps subst should return a closure, with some things precomputed?

The __str__ method of a pattern takes an optional argument, a list of strings to represent the variables.


Waffle
-----

All of the states involved can be expressed as coherent states, under the action of raising and lowering operators.  

The basic abstraction is a pattern, a kind of generalised vector or outer product.  Column patterns input an array of complex vectors |x|, and generate a block column vector whose |[i]| block is a function of |x[i].conjugate()|.  Row patterns are similar, but generate a row vector whose |[i]| block is a function of |x[i]|.  Matrix patterns input a list of left-hand vectors |x| and a list of right-hand vectors |y|; they generate a block array whose |[i, j]| block is a function of |x[i].conjugate()| and |y[j]|.  A matrix pattern will often be constructed as the outer product of a column pattern by a row pattern.

A sum pattern is similar to a row or column pattern, but represents the sum of the elements instead of a vector or matrix of them.

One kind of pattern represents matrices whose elements are polynomials of the inputs, or other ordinary complex expressions.  The other kind represents matrices of Dirac algebra objects, bras, kets, and operators.  Both kinds can be differentiated with respect to their inputs; there will be two derivative methods, to implement Wertinger calculus, but I haven't figured out the details.

There's an obvious way to generalise and simplify the State/Bra/Ket idea.  States and polynomials are elements, rows, columns, and outer products are forms.  A column of complex patterns is conjugated, a column of state patterns represents bras.  Note that outer products of bras and kets don't reduce to outer products of a polynomial in the bra parameters by one in the ket parameters.

It's tempting to represent a polynomial as a sum pattern of single or double sided monomials.  That won't work: the terms don't form a pattern.  So polynomials will have to be a distinct type.

In order to factor quantum manipulations out of inner loops, patterns partially evaluate themselves: an inner product of a bra and a ket will reduce to an arithmetic pattern in the parameters.

Presumably, derivatives still form row vectors, or column vectors for the conjugate derivative.  So we need identity patterns, that stack the parameters into vectors these can multiply.  The alternative is to have a separate tuple type for these, as in Spivak and Wisdom's notation.

With patterns, the parameters are stored separately from the bra or ket.  So there's no need for |prms|; the old |smrp| is replaced by |subst|.