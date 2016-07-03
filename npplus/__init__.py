# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Numpy and pyplot enhancements and alternatives.

npplus.basic
------------

Versions of concatenate and stack that broadcast their components:

``a_(a1, a2, ...)``
    Stack arrays along a new axis, but first broadcast them to the same
    shape.  By default the new axis=0, but accepts axis keyword argument.
``cat_(a1, a2, ...)``
    Join arrays along a given axis, but first broadcast them to the same
    shape except for that axis.  By default join along axis=0 (that is,
    the first axis of the array with the largest number of dimensions),
    but accepts axis keyword argument.

More workhorse array constructors:

``span(a, b, n)``
    Same as ``r_[a:b:n*1j]`` or ``linspace(a,b,n)``, except `a` and
    `b` may be conformable arrays.  The new axis=0 by default, but accepts
    axis keyword.
``spanl(a, b, n)``
    Same as ``logspace(log10(a),log10(b),n)``, except `a` and `b` may be
    conformable arrays.  The new axis=0 by default, but accepts axis
    keyword.  The `logspace` interface is rarely what you want.

Elementwise min, max, and abs functions arise frequently:

``max_(a1, a2, ...)``
    elementwise max with any number of conformable array-like arguments
``min_(a1, a2, ...)``
    elementwise min with any number of conformable array-like arguments
``abs_(a1, a2, ...)``
    elementwise ``linalg.norm`` with any number of conformable array-like
    arguments
``atan(y [,x])``
    combined ``arctan`` and ``arctan2`` with more convenient name

Rank preserving axis methods for finite difference operations, like
``diff``; all accept axis keyword:

``cum(x)``
    ``cumsum`` with prepended 0, an inverse of ``diff``
``zcen(x)``
    pairwise averages to go with ``diff`` (pairwise differences)
``pcen(x)``
    ``zcen``, but copy endpoints

npplus.pwpoly
-------------

Piecewise polynoimals are ubiquitous in practical numerical analysis.
The PwPoly class is inspired by the Polynomial class, providing
equivalent generic functionality for all piecewise polynomials.  There
is a PerPwPoly subclass for periodic piecewise polynomial functions.
Let ``pp`` and ``pq`` be piecewise polynomials, then:

``pp(x)``
    Piecewise polynomials are functions of a single variable x.  The
    input x may be any array_like, and pp(x) will have the same shape.
    A piecewise polynomial may also represent a curve in several
    dimensions, in which case pp(x) will have leading dimensions before
    x.shape.
``pp.xk, pp.c``
    The knot points and coefficients of the polynomials are available.
``pp+pq, pp-pq, pp*pq, -pp, 21*pp+7, pp/3``
    Arithmetic operators are defined.  If ``pp`` and ``pq`` do not have
    the same knot points, then the results of the binary operations will
    be piecewise polynomials with the union of the knot points.
``pp.deriv()``
    The derivative of a piecewise polynomial is a piecewise polynomial
    of one less degree with the same knot points.
``pp.integ()``
    The indefinite integral of a piecewise polynomial is a piecewise
    polynomial of one greater degree with the same knot points.
``pp.roots(value)``
    Return all `x` for which ``pp(x)==value``.

You usually construct piecewise polynomials using one of these four
functions:

``pline(x, y)``
    Construct the polyline passing through points ``(x,y)``.  The `y` may have
    leading dimensions for polylines through multidimensional spaces.
    By default ``pline(x,y)`` is constant before the first `x` or after
    the last `x`, but ``pline(x,y,1)``, which is a special case of the
    extrap keyword, extrapolates with the slope from the interior sides
    of both endpoints.
``plfit(xk, x, y)``
    Construct the polyline with knots at xk which is the best fit to
    the points ``(x,y)``.  The y may have leading dimensions for polylines
    through multidimensional spaces.  You may supply explicit standard
    deviations for the `y` points; by default all `y` are equally weighted.
``spline(x, y)``
    Construct a cubic spline passing through points ``(x,y)``.  The `y` may have
    leading dimensions for splines through multidimensional spaces.  A spline
    has maximumal smoothness at all interior knot points, that is, all
    except the final non-zero derivative are continuous.  You can get
    splines of other than cubic degree by supplying an explicit degree
    argument, for example, ``spline(x,y,4)`` produces a 4-th degree
    piecewise polynomial with its first three derivatives continuous.
    Other keyword arguments allow you to specify boundary conditions at
    either endpoint, and extrapolation beyond the endpoints.
``splfit(xk, x, y)``
    Construct the piecewise polynomial with knots at `xk` which is the best
    fit to the points ``(x,y)``.  The `y` may have leading dimensions for curves
    through multidimensional spaces.  You may supply explicit standard
    deviations for the `y` points; by default all `y` are equally weighted.
    Like spline, the default degree is 3, but you may supply any degree.
    The algorithm is chi2 minimization with continuity constraints on
    the function values and their derivatives at all the interior knots.
    By default, splfit produces the maximally continuous curve, that is,
    the spline curve.  However, you may specify continuity to any smaller
    degree.  With the ``cost=1`` keyword, `splfit` will return the Lagrange
    multipliers associated with all constraints, in addition to the
    piecewise polynomial.

Each of these functions accepts a per=1 keyword to return a periodic
piecewise linear function.

The PwPoly (and PerPwPoly) constructor is also occasionally useful:

``PwPoly(x, y, dydx, ...)``
    Construct the piecewise polynomial with knots at x passing through
    points ``(x,y)`` with derivatives `dydx`.  If you specify nd derivatives,
    the polynomials will be of degree ``2*nd+1``.  Beyond the first derivative,
    you must actually specify polynomial coefficients, that is, you specify
    ``d2ydx2/2, d3ydx3/6,`` and so on.  For a given degree deg, the piecewise
    polynomial is continuous only up to the ``(deg-1)/2``-th derivative at the
    knots.  The arguments to PwPoly define the polynomial within each
    interval of x according only to its values and derivatives at the
    endpoints the maximally local description.  Contrast this with the
    maximally smooth piecewise polynomials returned by spline.

npplus.lsqfit
-------------

Linear and non-linear least squares fitting are also ubiquitous in
numerical work.  The pwpoly module already includes the important
least squares fitting functions plfit and splfit.  The lsqfit module
provides more general model fitting functions:

``regress(data, m1, m2, ...)``
    Return the coefficients `p` of a linear model ``p[1]*m1+p[2]*m2+...``
    that best fit the given data in a least squares sense.  Each of the
    `m1`, `m2`, ... must be conformable with data.  The optional errs keyword
    allows you to specify standard deviation for each data point.  With
    the model=1 keyword, regress returns a ModelFit instance m providing
    more detailed information about the fit: ``m()`` is the best fit predicted
    data, ``m.p`` are the best fit coefficients, ``m.pcov`` are the covariances
    of `p`, and ``m.s`` and ``m.u`` are the singular values and p-vectors for
    the ``[m1, m2, ...]`` matrix.
``levmar(data, f, p0, args)``
    Returns a ModelFit instance m corresponding to the best fit of a
    parametrized non-linear function ``f(p,args)`` to the given data.  The
    args are the independent variables of the family of models, and p0 is
    the intial guess for `p` such that ``f(p,args) == data``.  The ModelFit
    instance can be called as a function ``m(args)`` to return ``f(m.p,args)``
    with `p` set to the best fit parameters ``m.p``.  The covariances
    of the best fit `p` are ``m.pcov``.  The levmar function accepts an `errs`
    keyword to specify standard deviations for the data points.

npplus.pcwise
-------------

A decorator to aid writing functions of one variable `x` which
have different algorithms in different domains of `x`::

    @pcwise
    def fun(x):
        def funlo(x):
            return 1. - x
        def funmid(x):
            return numpy.sin(x)
        def funhi(x):
            return x**2 - 1.
        return funlo, xa, funmid, xb, funhi
"""

from .basic import *
from .pwpoly import *
from .pcwise import *
from .lsqfit import *
