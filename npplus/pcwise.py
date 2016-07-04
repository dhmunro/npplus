# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""A decorator to neatly write piecewise functions.

The decorator is an aid to writing functions of one variable `x` which
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

Defines ``fun(x)`` to be ``funlo(x)`` for ``x<xa``, ``funmid(x)`` for
``xa<=x<xb``, and ``funhi(x)`` for ``xb<=x``.  Any number of domains
is allowed.

--------
"""

__all__ = ['pcwise']

from numpy import array, asarray, diff, zeros
from functools import wraps


def pcwise(f):
    """Decorator to simplify creating ``f(x)`` with algorithm dependent on `x`.

    Use the pcwise decorator like this::

        @pcwise
        def f(x):
            '''Return a function of a single real variable x.'''
            def f0(x):
                ...algorithm when       x < x1
            def f1(x):
                ...algorithm when x1 <= x < x2
            def f2(x):
                ...algorithm when x2 <= x < x3
            <and so on>
            return (f0, x1, f1, x2, f2, ...)

    Any of the `fN` in the return value may be a string to raise a
    ValueError with that string for any points in that range.
    """
    fxtuple = f(None)
    funcs = list(fxtuple[0::2])
    xvals = array(fxtuple[1::2])
    if len(funcs) != len(xvals)+1:
        raise TypeError("@pcwise function must return "
                        "(f0,x1,...xN,fN) sequence.")
    if xvals.size > 1 and (diff(xvals) <= 0).any():
        raise TypeError("@pcwise function must return "
                        "(f0,x1,...xN,fN) sequence with increasing xI.")
    for i, fn in enumerate(funcs):
        try:
            _ = fn + ''
        except TypeError:
            pass
        else:  # replace string by function that raises ValueError(fn)
            funcs[i] = _value_error_func(fn)
    funcs = tuple(funcs)

    @wraps(f)
    def multif(x):
        x = asarray(x)
        s = x.shape
        ialg = xvals.searchsorted(x)
        if not s:
            return funcs[ialg](x)
        result = zeros(s)
        for mask, f in [(ialg == i, fi) for i, fi in enumerate(funcs)]:
            xm = x[mask]
            if xm.size:
                result[mask] = f(xm)
        return result

    return multif


def _value_error_func(msg):

    def raise_value_error(x):
        raise ValueError(msg)

    return raise_value_error
