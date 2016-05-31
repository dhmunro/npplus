"""Yorick-like APIs with no good numpy equivalent

spanl(start, stop, num)   returns log-spaced points
max_(a, b, c, ...)   elementwise max with any number of args
min_(a, b, c, ...)   elementwise min with any number of args
abs_(a, b, c, ...)   elementwise linalg.norm with any number of args
atan(a [,b])         combined one or two argument atan
cum(x, axis=)        cumsum with prepended 0
zcen(x, axis=)       pairwise averages, like diff (pairwise differences)
pcen(x, axis=)       zcen, but copy endpoints
"""

__all__ = ['span', 'spanl', 'cat_', 'a_', 'max_', 'min_', 'abs_', 'atan',
           'cum', 'zcen', 'pcen']

from numpy import array, asarray, asanyarray, asfarray, zeros, zeros_like
from numpy import sign, log, absolute, log, exp, maximum, minimum, concatenate
from numpy import arctan, arctan2, pi, result_type
from numpy.linalg import norm

def span(start, stop, num=100, axis=0, dtype=None):
    """Return numbers with equal spacing between start and stop.

    Parameters
    ----------
    start : array_like
    stop : array_like, conformable with start
    num : int, optional (default 100)
    axis : int, optional
        If start and stop are not scalars, the position of the new axis
        in the result (first by default).
    dtype : dtype, optional
        Type of output array, default infers from start and stop.

    Returns
    -------
    samples : ndarray
        samples[0] == `start`, samples[num-1] == `stop`,
        with equal ratios between successive intervening values

    See Also
    --------
    spanl, linspace, arange, logspace
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
    shape = (slice(None),)*axis + (slice(1,None),)
    s[shape] += ds.cumsum(axis=axis)
    shape = shape[:axis] + (slice(-1,None),)
    s[shape] = stop   # eliminate roundoff error from final point
    return s.astype(dtype) if dtype else s

def spanl(start, stop, num=100, axis=0, dtype=None):
    """Return numbers with equal ratios (log spaced) between start and stop.

    Both start and stop may be negative, but they may not have
    opposite sign, nor may either be zero.

    Parameters
    ----------
    start : array_like
    stop : array_like, conformable with start
    num : int, optional (default 100)
    axis : int, optional
        If start and stop are not scalars, the position of the new axis
        in the result (first by default).
    dtype : dtype, optional
        Type of output array, default infers from start and stop.

    Returns
    -------
    samples : ndarray
        samples[0] == `start`, samples[num-1] == `stop`,
        with equal ratios between successive intervening values

    See Also
    --------
    span, logspace, linspace, arange
    """
    start, stop = asfarray(start), asfarray(stop)
    s, start = sign(start), absolute(start)
    stop = s * stop
    if (stop <= 0.).any():
        raise TypeError("start and stop must be non-zero and same sign")
    return s * exp(span(log(start), log(stop), num, dtype=dtype))

def cat_(*args, **kwargs):
    """concatenate arrays on one axis

    This is like np.concatenate, except that the input arrays are passed
    as multiple arguments rather than as a sequence in one argument, and,
    more importantly, they are broadcast to a common shape on all axes
    except the one being joined.

    Parameters
    ----------
    a1, a2, ... : array_like
        The arrays to be joined.  The arrays will be broadcast to a common
        shape over all axes except the one being joined.

    Keywords
    --------
    axis : int, optional
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
        dtype = t if dtype is None else result_type(dtype, t)
        ndim = max(ndim, a.ndim)
        alist.append(a)
    if ndim < 1: ndim = 1
    if axis<-ndim or axis>=ndim:
        ValueError("axis keyword is out of bounds")
    shape = array([(1,)*(ndim-a.ndim)+a.shape for a in alist])
    lens = shape[:,axis].copy()
    shape[:,axis] = lens.sum()
    result = zeros(shape.max(axis=0), dtype=dtype)
    i, leading = 0, ((slice(None),)*ndim)[:axis]
    for di, a in zip(lens, alist):  # second pass broadcasts into result
        result[leading+(slice(i,i+di),)] = a
        i += di
    return result

def a_(*args, **kwargs):
    """stack arrays on one axis

    This is like np.stack, except that the input arrays are broadcast to
    a common shape before stacking, so that they need only be conformable
    rather than exactly the same shape.

    Primary use cases:
        a_(2, 3, 5, ...)  # instead of
        array([2, 3, 5, ...])
        a_(0, [2, 3, 5])  # instead of
        array([zeros(3,dtype=int), [2, 3, 5])

    Parameters
    ----------
    a1, a2, ... : array_like
        The arrays to be joined.  The arrays will be broadcast to a common
        shape before being joined.

    Keywords
    --------
    axis : int, optional
        The axis for the new dimension in the result, by default axis=0,
        meaning the first axis of the result.

    Returns
    -------
    joined : ndarray
        The stacked array.  The shape is
    """
    axis = kwargs.pop('axis', 0)
    if kwargs:
        axis = list(kwargs.keys())
        raise TypeError("unrecognized keyword argument: "+axis[0])
    alist, dtype, ndim = [], None, 0
    for a in args:  # first pass collects shapes
        a = asanyarray(a)
        t = a.dtype
        dtype = t if dtype is None else result_type(dtype, t)
        ndim = max(ndim, a.ndim)
        alist.append(a)
    if axis<-ndim-1 or axis>ndim>0:
        raise ValueError("axis keyword is out of bounds")
    shape = array([(1,)*(ndim-a.ndim)+a.shape for a in alist])
    shape = tuple(shape.max(axis=0))
    if axis < 0: axis = ndim+1 + axis
    s, t = shape[:axis], shape[axis:]
    result = zeros(s+(len(alist),)+t, dtype=dtype)
    s = ((slice(None),)*ndim)[:axis]
    for i, a in enumerate(alist):  # second pass broadcasts into result
        result[s+(i,)] = a
    return result

def max_(a, *args):
    """Return elementwise maximum of any number of arguments."""
    for b in args:
        a = maximum(a, b)
    return a

def min_(a, *args):
    """Return elementwise minimum of any number of arguments."""
    for b in args:
        a = minimum(a, b)
    return a

# note this does more sqrt operations than needed
# problem with summing squares is overflow/underflow at half range
# problem with normalizing is lots of elementwise logic and that
#    the divides are about as expensive as sqrts
def abs_(a, *args):
    """Return elementwise 2-norm of any number of arguments."""
    if not args:
        return absolute(a)
    for b in args:
        a = norm((a,b), axis=0)
    return a

def atan(a, b=None, out=None, branch=None):
    """Return arctan with one argument, arctan2 with two arguments.

    Parameters
    ----------
    a : array_like
    b : array_like, optional
    out : ndarray of proper shape to hold result, optional
    branch : array_like, optional
        Branch cut angle (minimum value that can be returned).
        Ignored unless `b` is given.

    Results
    -------
    angle : ndarray
        The angle in radians whose tangent is `a` if `b` not given,
        or the angle from the ray `(1,0)` to the point `(b,a)` if
        `b` is given.

    With `branch`n two argument mode, `branch <= angle < branch+2*pi`.
    Default is essentially -pi, except that `atan(0,-1)` returns pi,
    but `atan(0,-1,branch=-pi)` returns -pi.  The most import case is
    arguably `branch=0`, which returns `0<=angle<2*pi` as expected.
    """
    if b is None:
        return arctan(a, out=out)
    a = arctan2(a, b, out=out)
    if branch is None:
        return a
    tp = 2.*pi
    # return tp - (branch-a)%tp + branch
    return (a-branch)%tp + branch

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
    cum : ndarray
        The cumulative sums, starting with 0.  The shape of the output
        is the same as `a`, except along `axis`, which is larger by 1.

    See Also
    --------
    cumsum, diff

    Examples
    --------
    >>> x = array([[1,1,1,1], [2,2,2,2]])
    >>> axismeth.cum(x)
    array([ 0,  1,  2,  3,  4,  6,  8, 10, 12])
    >>> axismeth.cum(x,axis=0)
    array([[0, 0, 0, 0],
           [1, 1, 1, 1],
           [3, 3, 3, 3]])
    >>> axismeth.cum(x,axis=1)
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
    """Zone center, computing means of adjacent elements along an axis.

    This is a companion to `diff`.  For example, given values `f` at
    sorted points `x`,
        sum(zcen(f) * diff(x))
    is the trapezoid-rule definite integral of f(x), and
        cum(zcen(f) * diff(x))
    is the indefinite integral.

    Parameters
    ----------
    a : array_like
    axis : int, optional
        The axis along which to operate.  Default -1 means apply to
        the final axis, like `diff`.

    Returns
    -------
    zcen : ndarray
        The zone centered values.  The shape of the output is the same
        as `a`, except along `axis`, which is smaller by 1.

    See Also
    --------
    diff, cum, pcen

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
    """Point center, computing adjacent means and leaving endpoints same.

    Parameters
    ----------
    a : array_like
    axis : int, optional
        The axis along which to operate.  Default -1 means apply to
        the final axis, like `zcen`.

    Returns
    -------
    pcen : ndarray
        The zone centered values.  The shape of the output is the same
        as `a`, except along `axis`, which is larger by 1.

    See Also
    --------
    zcen, diff

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
