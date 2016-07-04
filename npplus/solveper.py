# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Periodic variants of solve_banded.

Adds corner elements to banded matrix connecting final elements of `x`
to first equations and first elements of `x` to final equations.

Also include `solves_banded`, a variant of `solveh_banded` that does not
require a positive definite symmetric matrix.

--------
"""

from numpy import asfarray, result_type, zeros, eye, arange, roll, diag
from numpy import concatenate
from scipy.linalg import solve_banded, solve


def solve_periodic(l_and_u, ab, b, overwrite_ab=False, overwrite_b=False,
                   check_finite=True):
    """Variant of solve_banded which includes matrix corner elements.

    Layout of banded matrix `ab` is::

        ab[u+i-j, j] == a[i, j]
        ab[i, j] == a[i+j-u, j]

    Example (l,u)=(2,1) for 6x6::

        A50  a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55
        a10  a21  a32  a43  a54  A05
        a20  a31  a42  a53  A04  A15

        A50   0    0    x    x    x    copy last row before first
        a00  a01   0    0   A04  A05  <-- matrix begins here
        a10  a11  a12   0    0   A15
        a20  a21  a22  a23   0    0
         0   a31  a32  a33  a34   0
         0    0   a42  a43  a44  a45
        A50   0    0   a53  a54  a55  <-- matrix ends here
         x    x    0    0   A04  A05   copy first two rows after last
         x    x    x    0    0   A15

    See Also
    --------
    scipy.linalg.solve_banded : for description of parameters and returns

    Notes
    -----
    Outline of the algorithm:

    1. Solve `ab` without the corner elements using solve_banded.
    2. Solve `ab` without corner elements for ``b=0`` except 1 for each
       of the first `u` equations and last `l` equations, a total of ``u+l``
       solves with one more call to solve_banded.
    3. Construct a dense ``u+l`` square matrix and solve to find the
       linear combination of the results of (2) which compensates
       for the corner elements omitted in (1).
    """
    ab, b = asfarray(ab), asfarray(b)
    dtype = result_type(ab, b)
    n = ab.shape[1]  # number of equations == number of unknowns
    l, u = l_and_u   # number of lower, upper diagonals
    lpu = l + u      # < n (or solve_banded will fail)
    # first u variables, last u equations are the u corner (upper right of a)
    # last l variables, first l equations are the l corner (lower left of a)

    # construct l+u square matrix holding the corner matrix elements
    #  0   A04  A05    begin with the l lowers, first l equation corner
    #  0    0   A15
    # A50   0    0     then the u uppers, last u equation corner
    corn = zeros((lpu, lpu), dtype)
    if u:
        corn[-u:, :u] = _diag_to_norm((u-1, 0), ab[:u, :u])
    if l:
        corn[:l, -l:] = _diag_to_norm((0, l-1), ab[-l:, -l:])

    # solve equation without corner marix elements (ignoring them)
    xx = solve_banded(l_and_u, ab, b, overwrite_ab=False,
                      overwrite_b=overwrite_b, check_finite=check_finite)

    # solve equation l+u times with only first l and last u equations non-0
    bb = zeros((n, lpu), dtype)
    if u:
        bb[-u:, -u:] = eye(u, dtype=dtype)
    bb[:l, :l] = eye(l, dtype=dtype)
    xy = solve_banded(l_and_u, ab, bb, overwrite_ab=overwrite_ab,
                      overwrite_b=overwrite_b, check_finite=check_finite)

    # if ad is the diagonal part of a (excluding corners)
    # ad.dot(xx) = b   and   ad.dot(xy) = bb (0 except 1 in a single position)
    mask = roll(arange(n) < lpu, -l)  # first u, last l elements of x
    acx = corn.dot(xx[mask, ...])  # (last u & 1st l eqns, trailing axes of b)
    acy = corn.dot(xy[mask, :])  # (last u & 1st l eqns, 2nd axis of bb)
    # seek x such that a.dot(x) = b
    # seek p such that x = xx - xy.dot(p)
    # a.dot(x) = ad.dot(xx) + acx - p - acy.dot(p) = b
    # ==>  p + acy.dot(p) = acx    (lpu square dense solve finds p)
    corn = eye(lpu, dtype=dtype) + acy  # the matrix to be solved
    p = solve(corn, acx, overwrite_a=True, overwrite_b=True,
              check_finite=False)
    return xx - xy.dot(p)


def _diag_to_norm(l_and_u, ab):
    # Convert from diagonal ordered matrix form to normal matrix ordering.
    l, u = l_and_u
    bw, n = ab.shape
    if bw != u+1+l:
        raise ValueError("matrix shape inconsistent with given l and u")
    a = diag(ab[u])  # construct (n,n) matrix with main diagonal
    a1, np1 = a.ravel(), n+1
    for i, d in enumerate(reversed(ab[:u])):
        a1[i+1:n+n*(n-i-1):np1] = d[i+1:]  # fill in upper diagonals
    for i, d in enumerate(ab[u+1:]):
        a1[(i+1)*n::np1] = d[:-1-i]  # fill in lower diagonals
    return a


def solves_periodic(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
                    check_finite=True):
    """Variant of solve_periodic for symmetric matrices.

    Upper form unless keyword lower is false.  Upper and lower forms are the
    same as scipy.linalg.solveh_banded, except input to solves_periodic need
    not be positive definite, and the entire ab array is used::

        A40  A51  a02  a13  a24  a35   upper form
        A50  a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55

        a00  a11  a22  a33  a44  a55   lower form
        a10  a21  a32  a43  a54  A05
        a20  a31  a42  a53  A04  A15

    See Also
    --------
    scipy.linalg.solveh_banded : for description of parameters and returns
    solve_periodic : for more details
    """
    b, u = asfarray(b), ab.shape[0] - 1
    ab = _solves_symgen(ab, u, lower)
    # input overwrite_ab unused
    return solve_periodic((u, u), ab, b, overwrite_ab=True,
                          overwrite_b=overwrite_b, check_finite=check_finite)


def solves_banded(ab, b, overwrite_ab=False, overwrite_b=False, lower=False,
                  check_finite=True):
    """Variant of solve_banded for arbitrary symmetric matrices.

    Upper form unless keyword lower is false.  Upper and lower forms are the
    same as scipy.linalg.solveh_banded, except input to solves_periodic need
    not be positive definite, and the entire ab array is used::

        A40  A51  a02  a13  a24  a35   upper form
        A50  a01  a12  a23  a34  a45
        a00  a11  a22  a33  a44  a55

        a00  a11  a22  a33  a44  a55   lower form
        a10  a21  a32  a43  a54  A05
        a20  a31  a42  a53  A04  A15

    See Also
    --------
    scipy.linalg.solveh_banded : for description of parameters and returns

    Notes
    -----
    Use solveh_banded if the matrix is known to be positive definite.
    """
    b, u = asfarray(b), ab.shape[0] - 1
    ab = _solves_symgen(ab, u, lower)
    # input overwrite_ab unused
    return solve_banded((u, u), ab, b, overwrite_ab=True,
                        overwrite_b=overwrite_b, check_finite=check_finite)


def _solves_symgen(ab, u, lower):
    ab = asfarray(ab)
    if lower:
        a = ab[-1:0:-1].copy()
        for i, row in enumerate(a):
            row[:] = roll(row, u-i)
        return concatenate((a, ab), axis=0)
    else:
        a = ab[-2::-1].copy()
        for i, row in enumerate(a):
            row[:] = roll(row, -1-i)
        return concatenate((ab, a), axis=0)
