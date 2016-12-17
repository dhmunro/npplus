# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Enhancements for basic numpy functionality.

Enhancements fall into several categories:

1. Array building.  Provides ``a_(a1, a2, ...)`` and ``cat_(a1, a2, ...)``,
   like ``array([a1, a2, ...])`` and ``concatenate((a1, a2, ...))`` except
   that the arguments are broadcst to the required shapes if needed.
   Provides ``span(a,b,n)`` and ``spanl(a,b,n)`` like ``linspace(a,b,n)``
   and ``logspace(log10(a),log10(b),n)``, except that ``a`` and ``b``
   may be points in multidimensional space.  All npplus array building
   functions accept the ``axis=`` keyword.

2. Finite difference axis methods ``zcen``, ``cum``, and ``pcen`` to
   to supplement the ``cumsum``, ``cumprod``, and ``diff`` functions in
   numpy.  For example, ``cum(zcen(y)*diff(x))`` is the finite difference
   analog of the indefinite integral of y dx.

3. Redefine the ``range`` function in python 2 to be ``xrange`` so that
   it works the same as in python 3.  The python 3 way is a better choice,
   especially since numpy provides the ``arange`` function.

4. Provide multiple argument elementwise ``min_`` and ``max_`` functions.
   Also provide a multiple argument ``abs_`` function that gives Euclidean
   distance in multdimensional space.

5. Combine the one and two argument arctan in a single ``atan`` function.
   In two argument mode, provides for branch cut at any angle.
   Note that ``abs_(y,x)`` and ``atan(y,x)`` work well together.

--------
"""

__all__ = ['span', 'spanl', 'cat_', 'a_', 'max_', 'min_', 'abs_', 'atan',
           'cum', 'zcen', 'pcen', 'range']

import sys
if sys.version_info < (3,):
    range = xrange
else:
    range = sys.modules['builtins'].range

from numpy import array, asanyarray, asfarray, zeros, zeros_like
from numpy import sign, absolute, log, exp, maximum, minimum, concatenate
from numpy import arctan, arctan2, pi, sqrt


def span(start, stop, num=100, axis=0, dtype=None):
    """Return numbers with equal spacing between start and stop.

    Parameters
    ----------
    start,stop : array_like
        Shapes must be conformable but need not match exactly.
    num : int, optional
        Number of points in result.
    axis : int, optional
        If start and stop are not scalars, the position of the new axis
        in the result (default 0).
    dtype : dtype, optional
        Type of output array, by default infer from start and stop.

    Returns
    -------
    samples : ndarray
        ``samples[0] == start``, ``samples[num-1] == stop``,
        with equal differences between successive intervening values

    See Also
    --------
    spanl : equal ratio (log) spacing
    numpy.linspace : standard numpy function
    numpy.arange : standard numpy function
    """
    start, stop = asfarray(start), asfarray(stop)
    shape = zeros_like(start + stop)
    start, stop = start+shape, stop+shape
    shape = shape.shape
    if axis < 0:
        axis = axis + len(shape)+1
    shape = shape[:axis] + (1,) + shape[axis:]
    start, stop = start.reshape(shape), stop.reshape(shape)
    s = start.repeat(num, axis=axis)
    ds = ((stop-start)/(num-1)).repeat(num-1, axis=axis)
    shape = (slice(None),)*axis + (slice(1, None),)
    s[shape] += ds.cumsum(axis=axis)
    shape = shape[:axis] + (slice(-1, None),)
    s[shape] = stop   # eliminate roundoff error from final point
    return s.astype(dtype) if dtype else s


def spanl(start, stop, num=100, axis=0, dtype=None):
    """Return numbers with equal ratios (log spaced) between start and stop.

    Both start and stop may be negative, but they may not have
    opposite sign, nor may either be zero.

    Parameters
    ----------
    start,stop : array_like
        Shapes must be conformable but need not match exactly.
    num : int, optional
        Number of points in result.
    axis : int, optional
        If start and stop are not scalars, the position of the new axis
        in the result (default 0).
    dtype : dtype, optional
        Type of output array, by default infer from start and stop.

    Returns
    -------
    samples : ndarray
        ``samples[0] == start``, ``samples[num-1] == stop``,
        with equal ratios between successive intervening values

    See Also
    --------
    span : equal difference (linear) spacing
    numpy.logspace : standard numpy function
    """
    start, stop = asfarray(start), asfarray(stop)
    s, start = sign(start), absolute(start)
    stop = s * stop
    if (stop <= 0.).any():
        raise ValueError("start and stop must be non-zero and same sign")
    if axis:
        shape = s.shape
        if axis < 0:
            axis = axis + len(shape)+1
        s = s.reshape(shape[:axis] + (1,) + shape[axis:])
    s = s * exp(span(log(start), log(stop), num, axis))
    return s.astype(dtype) if dtype else s


def cat_(*args, **kwargs):
    """Concatenate arrays on one axis.

    This is like np.concatenate, except that the input arrays are passed
    as multiple arguments rather than as a sequence in one argument, and,
    more importantly, they are broadcast to a common shape on all axes
    except the one being joined.

    Parameters
    ----------
    a1,a2,... : array_like
        The arrays to be joined.  The arrays will be broadcast to a common
        shape over all axes except the one being joined.
    axis : int, optional keyword
        The axis along which to join the arrays, by default axis=0, meaning
        the first axis of the input with the maximum number of dimensions.

    Returns
    -------
    joined : ndarray
        The concatenated array.
    """
    axis = kwargs.pop('axis', 0)
    if kwargs:
        axis = list(kwargs.keys())
        raise TypeError("unrecognized keyword argument: "+axis[0])
    alist, dtype, ndim = [], None, 0
    for a in args:  # first pass collects shapes
        a = asanyarray(a)
        t = a.dtype
        if t.kind not in 'biufc':
            raise ValueError("cat_ only accepts numeric arrays")
        if dtype is None:
            dtype = array(0, dtype=t)
        else:
            dtype = dtype + array(0, dtype=t)
        ndim = max(ndim, a.ndim)
        alist.append(a)
    dtype = dtype.dtype
    if ndim < 1:
        ndim = 1
    if axis < -ndim or axis >= ndim:
        raise ValueError("axis keyword is out of bounds")
    shape = array([(1,)*(ndim-a.ndim)+a.shape for a in alist])
    lens = shape[:, axis].copy()
    shape[:, axis] = lens.sum()
    result = zeros(shape.max(axis=0), dtype=dtype)
    i, leading = 0, ((slice(None),)*ndim)[:axis]
    for di, a in zip(lens, alist):  # second pass broadcasts into result
        result[leading+(slice(i, i+di),)] = a
        i += di
    return result


def a_(*args, **kwargs):
    """Stack arrays on one axis.

    This is like np.stack, except that the input arrays are broadcast to
    a common shape before stacking, so that they need only be conformable
    rather than exactly the same shape::

        a_(2, 3, 5, ...)  # instead of
        array([2, 3, 5, ...])
        a_(0, [2, 3, 5])  # instead of
        array([zeros(3,dtype=int), [2, 3, 5])

    Parameters
    ----------
    a1,a2,... : array_like
        The arrays to be joined.  The arrays will be broadcast to a common
        shape before being joined.
    axis : int, optional keyword
        The axis for the new dimension in the result, by default axis=0,
        meaning the first axis of the result.

    Returns
    -------
    joined : ndarray
        The stacked array.
    """
    axis = kwargs.pop('axis', 0)
    if kwargs:
        axis = list(kwargs.keys())
        raise TypeError("unrecognized keyword argument: "+axis[0])
    alist, dtype, ndim = [], None, 0
    for a in args:  # first pass collects shapes
        a = asanyarray(a)
        t = a.dtype
        if t.kind not in 'biufc':
            raise ValueError("a_ only accepts numeric arrays")
        if dtype is None:
            dtype = array(0, dtype=t)
        else:
            dtype = dtype + array(0, dtype=t)
        ndim = max(ndim, a.ndim)
        alist.append(a)
    dtype = dtype.dtype
    if axis < -ndim-1 or axis > ndim > 0:
        raise ValueError("axis keyword is out of bounds")
    shape = array([(1,)*(ndim-a.ndim)+a.shape for a in alist])
    shape = tuple(shape.max(axis=0))
    if axis < 0:
        axis = ndim+1 + axis
    s, t = shape[:axis], shape[axis:]
    result = zeros(s+(len(alist),)+t, dtype=dtype)
    s = ((slice(None),)*ndim)[:axis]
    for i, a in enumerate(alist):  # second pass broadcasts into result
        result[s+(i,)] = a
    return result


def max_(a, *args):
    """Return elementwise maximum of any number of array-like arguments."""
    for b in args:
        a = maximum(a, b)
    return a


def min_(a, *args):
    """Return elementwise minimum of any number of array-like arguments."""
    for b in args:
        a = minimum(a, b)
    return a


def abs_(a, *args):
    """Return elementwise 2-norm of any number of array-like arguments.

    See Also
    --------
    numpy.linalg.norm : norm along one axis of a single array
    """
    if not args:
        return absolute(a)
    # note this does more sqrt operations than needed
    # problem with summing squares is overflow/underflow at half range
    # problem with normalizing is lots of elementwise logic and that
    #    the divides are about as expensive as sqrts
    a = a_(a, *args)
    return sqrt((a.conj() * a).sum(axis=0))


def atan(a, b=None, out=None, branch=None):
    """Return arctan with one argument, arctan2 with two arguments.

    Parameters
    ----------
    a : array_like
    b : array_like, optional
    out : ndarray of proper shape to hold result, optional
    branch : array_like, optional
        Branch cut angle, the minimum value that can be returned.
        Ignored unless `b` is given.

    Returns
    -------
    ndarray
        The angle in radians whose tangent is `a` if `b` not given,
        or the angle from the ray ``(1,0)`` to the point ``(b,a)`` if
        `b` is given.

    Notes
    -----
    In two argument mode, ``branch <= angle < branch+2*pi``.  Default
    is essentially ``-pi``, except that ``atan(0,-1)`` returns ``pi``,
    but ``atan(0,-1,branch=-pi)`` returns ``-pi``.  The most import
    case is arguably ``branch=0``, which returns ``0<=angle<2*pi`` as
    expected.
    """
    if out is None:
        if b is None:
            return arctan(a)
        a = arctan2(a, b)
    else:  # these fail in python3 when out is None
        if b is None:
            return arctan(a, out=out)
        a = arctan2(a, b, out=out)
    if branch is None:
        return a
    # return 2pi - (branch-a)%2pi + branch
    a = (a-branch)%(2.*pi) + branch
    if out is not None:
        out[...] = a
    return a

# Following three are finite difference companions to np.diff


def cum(a, axis=-1):
    """Calculate cumulative sums (cumsum) with prepended 0.

    `cum` is an inverse of `diff`, the finite difference analog of
    integration if `diff` is the analog of differentiation.

    Parameters
    ----------
    a : array_like
    axis : int, optional
        The axis along which to accumulate.  Default -1 means apply to
        the final axis, like `diff`.

    Returns
    -------
    ndarray
        The cumulative sums, starting with 0.  The shape of the output
        is the same as `a`, except along `axis`, which is larger by 1.

    See Also
    --------
    numpy.cumsum : same except missing leading 0
    numpy.diff : pairwise differences
    zcen : pairwise means

    Examples
    --------
    >>> x = array([[1,1,1,1], [2,2,2,2]])
    >>> axismeth.cum(x,axis=None)
    array([ 0,  1,  2,  3,  4,  6,  8, 10, 12])
    >>> axismeth.cum(x,axis=0)
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [3, 3, 3, 3]])
    >>> axismeth.cum(x)
    array([[0, 1, 2, 3, 4],
           [0, 2, 4, 6, 8]])
    """
    a = asanyarray(a)
    if not a.shape:
        raise TypeError("Cannot apply cum to a scalar value.")
    if axis is None:
        a = a.ravel()
        z = [1]
    else:
        z = list(a.shape)
        z[axis] = 1
    return concatenate((zeros(z, dtype=a.dtype), a.cumsum(axis=axis)),
                       axis=axis)


def zcen(a, axis=-1):
    """Zone center by computing means of adjacent elements along an axis.

    This is a companion to `diff`.  For example, given values `f` at
    sorted points `x`, you can compute the definite and indefinite
    trapezoid rule integrals with::

        sum(zcen(f) * diff(x))  # definite integral
        cum(zcen(f) * diff(x))  # indefinite integral

    Parameters
    ----------
    a : array_like
    axis : int, optional
        The axis along which to operate.  Default -1 means apply to
        the final axis, like `diff`.

    Returns
    -------
    ndarray
        The zone centered values.  The shape of the output is the same
        as `a`, except along `axis`, which is smaller by 1.

    See Also
    --------
    numpy.diff : pairwise differences
    cum : cumulative sums starting from 0
    pcen : point center

    Examples
    --------
    >>> x = array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> axismeth.zcen(x)
    array([[ 1.5,  2.5,  3.5],
           [ 5.5,  6.5,  7.5]])
    >>> axismeth.zcen(x,axis=0)
    array([[ 3.,  4.,  5.,  6.]])
    """
    # see lib/function_base.py: diff
    a = asanyarray(a)
    if not a.shape:
        raise TypeError("Cannot zone center a scalar value.")
    if a.shape[axis] < 2:
        raise TypeError("Cannot zone center an axis with only one element.")
    slice1 = [slice(None)] * a.ndim
    slice2 = list(slice1)
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    # note implicit promotion to float type
    return (a[tuple(slice1)] + a[tuple(slice2)]) * 0.5


def pcen(a, axis=-1):
    """Point center by computing adjacent means and leaving endpoints same.

    Parameters
    ----------
    a : array_like
    axis : int, optional
        The axis along which to operate.  Default -1 means apply to
        the final axis, like `zcen`.

    Returns
    -------
    ndarray
        The zone centered values.  The shape of the output is the same
        as `a`, except along `axis`, which is larger by 1.

    See Also
    --------
    numpy.diff : pairwise differences
    zcen : zone center

    Examples
    --------
    >>> x = array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> axismeth.pcen(x)
    array([[ 1. ,  1.5,  2.5,  3.5,  4. ],
           [ 5. ,  5.5,  6.5,  7.5,  8. ]])
    >>> axismeth.pcen(x,axis=0)
    array([[ 1.,  2.,  3.,  4.],
           [ 3.,  4.,  5.,  6.],
           [ 5.,  6.,  7.,  8.]])
    """
    a = asanyarray(a)
    if not a.shape:
        raise TypeError("Cannot point center a scalar value.")
    s1 = [slice(None)] * a.ndim
    s2 = list(s1)
    s1[axis] = slice(0, 1)      # [0:1]
    s2[axis] = slice(-1, None)  # [-1:]
    if a.shape[axis] < 2:
        return concatenate((a[tuple(s1)], a[tuple(s2)]), axis=axis)
    slice1, slice2 = list(s1), list(s1)
    slice1[axis] = slice(1, None)   # [1:]
    slice2[axis] = slice(None, -1)  # [:-1]
    return concatenate((a[tuple(s1)],
                        (a[tuple(slice1)] + a[tuple(slice2)]) * 0.5,
                        a[tuple(s2)]), axis=axis)
