this is a preliminary implementation of my PhD project, where optimisation is done by solving normal equations.  it is unlikely to work well enough; the polykets project has the libraries required to run more stable algorithms on infinite matrices.

This library tests and profiles subprograms to calcuate the H and V matrices for a quartic oscillator.  It assumes the interface comes in three Python routines.  Each of these takes one argument, an linear combination of N coherent states over M-1 modes, represented as an N by M ndarray.  The first mode is a logarithmic weight and phase.

The first function, rho, should calculate a matrix of brackets between the states in the superposition.  The second, jsq, calculates the squared Jacobian matrix V.  The third, jham, computes the bracket of the Hamiltonian between the Jacobian vector and the linear combination.  Jsq and jham take the matrix returned by rho as a second argument.

The test system has three routines, test_rho, test_jsq, and test_jham.  Each of these takes a set of routines to test, and a generator to produce test superpositions.  The test functions return if everything works, and raise an exception if the routines fail some invariant that they should satisfy.

The unit plot.py generates Metapost plots of ensemble amplitudes.  plot.mp contains most of the Metapost code; it is included in the code generated.