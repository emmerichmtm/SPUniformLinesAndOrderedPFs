Replicate and verify the finite-front Solow--Polasky examples.

This script reproduces the two numerical examples in the report:

1. A dense connected front f2 = 1 - f1^2 with n=70 and k=10.
2. A disconnected ZDT3 Pareto front sample with n=100 and k=20.

It also performs two verification steps:

A. A beta sanity check for the ZDT3 example, comparing beta=1 and beta=2.
B. Several small brute-force checks comparing the dynamic-programming solution
   against complete enumeration on small ordered line instances.

The dynamic program uses the finite-line formula for Solow--Polasky diversity
under the exponential kernel on an ordered l1 front:

    D_beta(S) = 1 + sum_r tanh(beta * gap_r / 2),

where gap_r are consecutive distances in the induced ordered line coordinate.
The script uses only the Python standard library.
