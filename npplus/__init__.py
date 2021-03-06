# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Numpy and pyplot enhancements and alternatives.

*   A piecewise polynomial class :class:`npplus.pwpoly.PwPoly`, a more
    practical alternative to the :class:`scipy.interpolate.PPoly`.  A
    `PwPoly` instance `p` is naturally callable with ``p(x)``
    returning the value of the piecewise polynomial function.  You can
    combine `PwPoly` instances `p` and `q` with arithmetic operations,
    so that ``3*p*q - p/2 + 0.5`` is a new `PwPoly`.  Integration and
    differentiation ``p.integ()`` and ``p.deriv()`` also return new
    `PwPoly` instances.  A `PwPoly` may also represent a curve in
    multi dimensional space, so that ``p(x)`` has leading dimensions
    before the dimensions of `x`; in that case, ``p[i]`` is also a
    `PwPoly`.  For scalar `PwPoly` instances, ``p.roots(value)``
    returns the list of all `x` such that ``p(x) == value``.

*   Provides `spline` and `splfit` functions returning `PwPoly` instances:

    ``spline(x, y)``, see :func:`npplus.pwpoly.spline`
        The natural cubic spline through points ``(x, y)``.  You can specify
        other boundary conditions and any degree polynomials.
    ``splfit(xk, x, y)``, see :func:`npplus.pwpoly.splfit`
        The cubic spline with knot points at `xk`, which is the least
        squares best fit to points ``(x, y)``.  You can specify boundary
        conditions and any degree polynomials.  Optionally returns the
        Lagrange multipliers for all continuity abd boundary constraints.

    Unlike the :mod:`scipy.interpolate` spline functions, these do not use
    the compiled fitpack functions, only numpy and the scipy solve_banded
    functions.  You can mine this code if you have a variant problem.
    Both functions accept a ``per=1`` keyword to return periodic piecewise
    linear functions.  Both functions accept multi-dimensional `y` values
    to return curves in multi-dimensional space.  The `splfit` function
    allows you to specify standard deviations for the `y` values.

*   Variants :func:`npplus.pwpoly.pline` and :func:`npplus.pwpoly.plfit`
    with simplified arguments for the important special case of piecewise
    linear polylines.

*   Simple interfaces for linear and non-linear least squares fitting:

    ``regress(data, m1, m2, ...)``, see :func:`npplus.lsqfit.regress`
        Return the coefficients `p` of a linear model ``p[1]*m1+p[2]*m2+...``
        that best fit the given data in a least squares sense.  Each of the
        `m1`, `m2`, ... must be conformable with data.  Optionally returns
        convariances and other fit statistics.
    ``levmar(data, f, p0, args)``, see :func:`npplus.lsqfit.levmar`
        Returns a callable `m` such that ``m(args)`` is the best fit of a
        parametrized non-linear function ``f(p,args)`` to the given data.
        The `args` are the independent variables of the family of models,
        and `p0` is the intial guess for `p` such that ``f(p,args) ~ data``.
        The best fit parameters themselves are `m.p`, and other methods
        of `m` provide a complete statistical description of the fit.

*   A decorator :func:`npplus.pcwise.pcwise` to aid writing functions of
    one variable `x` which have different algorithms in different
    domains of `x`::

        @pcwise
        def fun(x):
            def funlo(x):
                return 1. - x
            def funmid(x):
                return numpy.sin(x)
            def funhi(x):
                return x**2 - 1.
            return funlo, xa, funmid, xb, funhi

*   Versions of `stack` and `concatenate` that broadcast their components,
    a generalized `linspace`, and a repaired `logspace`:

    ``a_(a1, a2, ...)``, see :func:`npplus.basic.a_`
        Stack arrays along a new axis, broadcasting first if needed.
    ``cat_(a1, a2, ...)``, see :func:`npplus.basic.cat_`
        Join arrays along a given axis, brodcasting first if needed.
    ``span(a, b, n)``, see :func:`npplus.basic.span`
        Like ``linspace(a,b,n)``, except `a` and `b` may have any conformable
        shapes to generate a line in multi-dimensional space.
    ``spanl(a, b, n)``, see :func:`npplus.basic.spanl`
        Like ``logspace(log10(a),log10(b),n)``, except `a` and `b` may be
        have any conformable shapes.

*   Elementwise min, max, and abs with any number of arguments:

    ``max_(a1, a2, ...)``, see :func:`npplus.basic.max_`
        Elementwise max with any number of conformable array-like arguments.
    ``min_(a1, a2, ...)``, see :func:`npplus.basic.min_`
        Elementwise min with any number of conformable array-like arguments.
    ``abs_(a1, a2, ...)``, see :func:`npplus.basic.abs_`
        Elementwise ``linalg.norm`` with any number of conformable array-like
        arguments.

*   Rank preserving axis methods for finite difference operations to
    supplement ``diff``.  For example, ``cum(zcen(y)*diff(x))`` is a finite
    difference indefinite integral:

    ``cum(x)``, see :func:`npplus.basic.cum`
        ``cumsum`` with prepended 0, an inverse of ``diff``
    ``zcen(x)``, see :func:`npplus.basic.zcen`
        pairwise averages to go with ``diff`` (pairwise differences)
    ``pcen(x)``, see :func:`npplus.basic.pcen`
        ``zcen``, but copy endpoints

*   An attribute-dict class :class:`npplus.itemattr.ADict`.  An `ADict`
    instance `ad` is a mapping whose items can be accessed either
    as items ``ad['name']`` or as attributes ``ad.name``.

*   A :func:`npplus.interactive.reloadx` function to simplify debugging
    a module in an interactive session.  Your workflow becomes a loop
    of edit source, `reloadx`, and `pdb` run or post-mortem without
    any external IDE required.

*   Wrappers for `pyplot` plotting functions like `plot` which return
    unwanted objects and clutter interactive terminals.  It is easier to
    type ``plt.plot`` in the rare case you want the object, rather than
    ``_=plot`` every time you don't.

*   A simple presentation-ready matplotlib style.

*   A module :mod:`npplus.pyplotx.interactive` you can import in
    `PYTHONSTARTUP` that gives you the `pylab` interactive environment
    plus all the `npplus` features.

--------

"""

from .basic import *  # noqa
from .pwpoly import *  # noqa
from .itemattr import *  # noqa
from .pcwise import *  # noqa
from .lsqfit import *  # noqa
