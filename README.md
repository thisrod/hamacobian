this is a preliminary implementation of my PhD project, where optimisation is done by solving normal equations.  it is unlikely to work well enough; the polykets project has the libraries required to run more stable algorithms on infinite matrices.

This library tests and profiles subprograms to calcuate the H and V matrices for a quartic oscillator.  It assumes the interface comes in three Python routines.  Each of these takes one argument, an linear combination of N coherent states over M-1 modes, represented as an N by M ndarray.  The first mode is a logarithmic weight and phase.

The first function, rho, should calculate a matrix of brackets between the states in the superposition.  The second, jsq, calculates the squared Jacobian matrix V.  The third, jham, computes the bracket of the Hamiltonian between the Jacobian vector and the linear combination.  Jsq and jham take the matrix returned by rho as a second argument.

The test system has three routines, test_rho, test_jsq, and test_jham.  Each of these takes a set of routines to test, and a generator to produce test superpositions.  The test functions return if everything works, and raise an exception if the routines fail some invariant that they should satisfy.

The unit plot.py generates Metapost plots of ensemble amplitudes.  plot.mp contains most of the Metapost code; it is included in the code generated.

State objects
----------

An object of class |KetRow| represents a row vector of kets.  The length of these objects is the number of kets; teh |wid| (width) function returns the number of parameters of each ket, which is one for the weight, plus the number of state parameters.

A KetSum represents a state that is the sum of these kets.  It is a ket when on the right of a product, a bra when on the left.

If the individual kets can be represented, iterating over the state returns representations of each component.

When |KetRow|s are multiplied, the one on the left has its Hermitian conjugate taken.  This product is not associative, so pay attention to brackets!  This is implemented by double dispatch.

Taking the deriviative of a |KetRow| returns a vector of derivatives wrt the parameters in the standalone constructor function (not the class constructor).

Adding a 1D array to a state advances the parameters by the array elements.  As usual, the parameter order comes from the standalone constructor.


Numerical experiments
------------------

|fixstep.py| demonstrates the failure of ensembles of coherent states to converge to a target state, using Levenberg-Marquardt, except in trivial cases.

|fock.py| solves linear least squares problems in a finite dimensional Fock space, and compares the results to solutions using the normal equations in the tangent space to G<sup>R</sup>.

Numpy style
----------

Everything is an |ndarray|; vectors are 1-dimensional; matrix products use dot.