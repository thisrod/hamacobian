Design of the states and patterns libraries
======

This file describes the libraries that implement simulations by variation of paths.  The theory behind the method, and its numerical stability, are described elsewhere.

The variation of paths method uses sums of coherent states and their derivatives with respect to coefficients and coherent amplitudes.  It requires the calculation of brackets of these, and moments of the Hamiltonian between them.  This calculation is repeated at every timestep of a simulation.

All of the states involved can be expressed as coherent states, under the action of raising and lowering operators.  

The basic abstraction is a pattern, a kind of generalised vector or outer product.  Column patterns input an array of complex vectors |x|, and generate a block column vector whose |[i]| block is a function of |x[i].conjugate()|.  Row patterns are similar, but generate a row vector whose |[i]| block is a function of |x[i]|.  Matrix patterns input a list of left-hand vectors |x| and a list of right-hand vectors |y|; they generate a block array whose |[i, j]| block is a function of |x[i].conjugate()| and |y[j]|.  A matrix pattern will often be constructed as the outer product of a column pattern by a row pattern.

A sum pattern is similar to a row or column pattern, but represents the sum of the elements instead of a vector or matrix of them.

One kind of pattern represents matrices whose elements are polynomials of the inputs, or other ordinary complex expressions.  The other kind represents matrices of Dirac algebra objects, bras, kets, and operators.  Both kinds can be differentiated with respect to their inputs; there will be two derivative methods, to implement Wertinger calculus, but I haven't figured out the details.

There's an obvious way to generalise and simplify the State/Bra/Ket idea.  States and polynomials are elements, rows, columns, and outer products are forms.  A column of complex patterns is conjugated, a column of state patterns represents bras.  Note that outer products of bras and kets don't reduce to outer products of a polynomial in the bra parameters by one in the ket parameters.

It's tempting to represent a polynomial as a sum pattern of single or double sided monomials.  That won't work: the terms don't form a pattern.  So polynomials will have to be a distinct type.

In order to factor quantum manipulations out of inner loops, patterns partially evaluate themselves: an inner product of a bra and a ket will reduce to an arithmetic pattern in the parameters.

Presumably, derivatives still form row vectors, or column vectors for the conjugate derivative.  So we need identity patterns, that stack the parameters into vectors these can multiply.  The alternative is to have a separate tuple type for these, as in Spivak and Wisdom's notation.

With patterns, the parameters are stored separately from the bra or ket.  So there's no need for |prms|; the old |smrp| is replaced by |subst|.