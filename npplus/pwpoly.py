# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Piecewise polynomial interpolation and curve fitting.

This module simplifies, replaces, or enhances many scipy.interpolate
programming interfaces.  The interface consists of two classes,
`PwPoly` and its periodic subclass `PerPwPoly`, and four functions,
`pline`, `spline`, `plfit`, `splfit`.  The `pline` and `plfit`
functions are special cases of the general `spline` and `splfit`
functions returning piecewise linear functions; `spline` and `splfit`
return piecewise cubics by default, but can be used to construct
piecewise polynomials of any degree.

The `spline` and `pline` functions return the interpolating function
passing exactly through a given set of points.  The `splfit` and
`plfit` functions return the least squares best fit to a cloud of
given points with given knot points (the `x` values separating the
polynomial pieces).  All four functions work for curves in
multidimensional space.  A quick tour of PwPoly functionality::

    f = pline(x, y)  # polyline through points (x, y)
    g = spline(x, y)  # natural cubic spline through (x, y)
    f = plfit(x, xdat, ydat)  # best fit polyline with knots at x
    g = splfit(x, xdat, ydat)  # best fit cubic spline with knots at x
    f(xp)  # if f from pline, same as interp(xp, x, y)
    g(xp)  # evaluate cubic spline at points xp
    h = 2*f + g  # operators +, -, * work for piecewise polynomials
    xv = h.roots(value)  # all xv such that h(xv) = value
    dydx = h.deriv()  # piecewise poly of one less degree than h
    iydx = h.integ()  # piecewise poly of one greater degree than h

The only common use cases for the PwPoly constructor are for making
histograms, that is, piecewise constant functions, which you do by
supplying one more or less y value than x value, and for constructing
piecewise cubic functions passing through ``(x,y)`` with given ``dy/dx``::

    histo = PwPoly(x, y)  # y.size = x.size+1 or x.size-1
    cubic = PwPoly(x, y, dydx)  # x, y, dydx same size

The spline interpolation and fitting routines here use only standard
numpy plus the scipy solve_banded function.  The scipy.interpolate
spline routines are wrappers around the old Fortran fitpack package.
A key advantage of the pwpoly module is that you can mine the code here
and adapt it to slightly different problems, whereas compiled fitpack
code is not adaptable.

--------
"""

__all__ = ['PwPoly', 'PerPwPoly', 'spline', 'pline', 'splfit', 'plfit']

from numpy import array, asarray, asfarray, zeros, zeros_like, ones, arange
from numpy import promote_types, eye, concatenate, searchsorted, einsum, roll
from numpy import newaxis, maximum, minimum, absolute, any, isreal, real
from numpy import prod, isclose, transpose, ones_like
from numpy import bincount, array_equal, sort
from numpy.linalg import inv, eigvals
from scipy.linalg import solve_banded, solveh_banded

from .solveper import solve_periodic, solves_periodic, solves_banded


class PwPoly(object):
    """Piecewise polynomial function.

    Typical usage::

        pwp = PwPoly(xk, yk, dydxk)  # define a piecewise cubic function
        y = pwp(x)                   # evaluate function at x
        y, dydx, d2ydx2 = pwp(x, 2)  # evaluate function and 2 derivatives

    The points xk are called "knot points" of the piecewise
    polynomial.  Outside the endpoints of xk, the function will have
    the degree to which it is continuous at the other endpoints.

    Parameters
    ----------
    xk : array_like
        List of abcissa values bounding the pieces, in increasing order.
        As a convenience, strictly decreasing `xk` are also accepted.

    yk,dydxk,d2ydx2k/2,... : array_like
        Each is a list of function and derivative values `yk`,
        `dydxk`, `d2ydx2k/2`, etc. corresponding to the points `xk`.
        This produces an odd degree polynomial in each interval,
        continuous to the specified derivatives at the xk points.

        Note that when you specify derivatives higher than first, you
        need to divide the n-th derivative by n!.  That is, you are
        actually furnishing the Taylor series coefficients, not the
        derivatives.

        To get even degree polynomials, specify the final derivative
        with one fewer point than `xk` -- this produces a polynomial
        of degree ``P=2*N`` continuous at `xk` up to the N-1
        derivatives you specified at all xk points, and with the P-th
        (not N-th) derivative/p!  equal to the final values specified.

        Finally, for complete generality, you can specify every
        argument with one more point than `xk`.  In this case, the
        arguments simply become the coefficients of the polynomials in
        each interval with no continuity guaranteed at the `xk`.  Note
        that the origin for coefficients in each interval is the first
        point of the interval (the smaller), except for the
        semi-infinite interval before the first knot point, which is
        relative to the first knot.

        All the `yk`, `dydxk`, etc. may have a set of leading axes to
        define a multidimensional piecewise polynomial function of `xk`.
        That is, the interpolation direction is the last axis of `xk`,
        `yk`, `dydxk`, etc.

    Attributes
    ----------
    xk : ndarray
        1D array of knot points in strictly increasing order.
    c : ndarray
        Coefficient array.  Axes are ``(degree, dimensionality,
        knots+1)`` The polynomial coefficients are for the polynomial
        in ``(x-xk[i-1])``, where ``xk[i-1] <= x < xk[i]``, except for
        ``c[...,0]``, which like ``c[...,1]`` is for the polynomial in
        ``(x-xk[0])``.  Coefficients are in order of increasing powers
        of `x`.

    See Also
    --------
    spline : PwPoly interpolation by cubic and other degree splines
    splfit : PwPoly fitting by cubic and other degree splines
    pline : polyline interpolation
    plfit : polyline fitting

    Notes
    -----
    Let p and q be PwPoly instances.  The following operators work:

    ``p(x)``
        Evaluate `p` at `x`, returning array of same shape as `x`.
        If `p` is multidimensional, its additional dimensions are leading
        dimensions of the result.
    ``p+q, p-q, -p, p*q``
        Return a new PwPoly instance.  If `p` and `q` have different
        `xk`, result the union of the two `xk`.  Either `p` or `q` may
        be scalars (or arrays same shape as ``p(0)``).  For array_like
        `q`, division `p/q` also works.
    ``p[i]``
        Components of `p`, only for multidimensional `p`.
    ``len(p)``
        First dimension length of `p`, only for multidimensional `p`.

    The PwPoly constructor produces a smooth fit that is local, in the
    sense that the function in the interval between consecutive knot
    points depends only on the given function and derivative values at
    the interval endpoints.

    Use the spline function to construct a PwPoly which is smoother
    (for a given degree) by using the function values you provide at
    all the knot points to determine the function within each
    interval.  The splfit function constructs a PwPoly that is the
    statistical best fit to a cloud of data points you provide.

    The pline and plfit functions are piecewise linear variants of
    spline and splfit.

    PwPoly is intended for low degree polynomials, usually degree 1
    and 3.  It should work tolerably well at or below degree 7, but
    roundoff errors will become significant at higher degree.  To get
    a better fit with PwPoly, use more knots, not higher degree.

    ``PwPoly(x, y)`` where `y` has one more or one less element than
    `x` produces a histogram, that is, a piecewise polynomial of
    degree zero.  This is probably the only useful application of the
    PwPoly constructor to produce a PwPoly of even degree.  For other
    even degree piecewise polynomials, you probably want to use spline
    or splfit.
    """
    def __init__(self, xk=None, *args):
        if xk is None:
            return
        xk = asfarray(xk)
        if xk.ndim != 1:
            raise TypeError("xk must be 1D array_like")
        nk = xk.size
        if nk > 1 and xk[-1] < xk[0]:
            # as a convenience, reverse xk and args so xk increasing
            xk = xk[::-1]
            args = asarray(args)[..., ::-1]
        argz = asfarray(args[-1])
        even = (argz.shape[-1] == nk-1)
        if even:
            args = args[0:-1]
            if not args:
                zero = zeros_like(argz[..., 0:1])
                args = (concatenate((zero, argz, zero), axis=-1), )
                even = False
        args = asfarray(array(args))  # copy of some float type
        nda = args.ndim - 2  # additional dimensions of args
        if nda < 0:
            raise TypeError("yk and derivatives must be at least 1D")
        dtype = promote_types(xk.dtype, args.dtype)
        if args.shape[-1] != nk+1:
            if args.shape[-1] != nk:
                raise TypeError("yk trailing axis inconsistent with xk size")
            if nk < 2:
                raise TypeError("xk needs at least 2 points when yk same size")
            # convert args from derivatives to polynomial coefficients
            h = args.shape[0] - 1
            fact = maximum(arange(h+1, dtype=dtype), 1).cumprod()
            args *= 1./fact.reshape((h+1,)+(1,)*(nda+1))
            if even:
                argz *= 1./(fact[0] * arange(h+1, 2*h+1, dtype=dtype).prod())
            # compute coefficients
            dx = (xk[1:] - xk[:-1]).reshape((1,)*nda + (nk-1,))
            left, rght = args[..., :-1], args[..., 1:]
            argz = argz[newaxis] if even else left[0:0, ...]
            c = concatenate((left, zeros_like(left), argz))
            rght = rght - polyddx(c, dx, h)
            dxn = dx + zeros((h+1,)+dx.shape)
            dxn[0] = 1
            dxn = dxn.cumprod(axis=0)  # [1, dx, dx**2, ..., dx**h]
            rght *= dxn  # normalize to 0<dx'<1
            m = self._two_sided_inverse(h)
            rght = einsum('ij,j...->i...', m, rght)  # now h+1...2*h+1 coeffs
            dxn *= dx*dxn[-1]  # [dx**(h+1), ..., dx**(2*h+1)]
            rght *= 1./dxn
            c = concatenate((left, rght, argz))
            # add extrapolation coefficients - just the final given ones
            left = zeros_like(c[..., 0:1])
            c = concatenate((left, c, left), axis=-1)
            c[0:h+1, ..., 0:nk+1:nk] = args[..., 0:nk:nk-1]
            args = c
        # set up for __call__
        self._rawinit(xk, args)  # will always concatenate and copy xk

    def _rawinit(self, xk, c):
        # permits setting xk, c without any copy
        if xk.size == c.shape[-1]:
            self.xk0 = xk
        else:
            self.xk0 = concatenate((xk[0:1], xk))
        self.xk = self.xk0[1:]
        self.c = c
        # treat xk0, c as if immutable
        # not worth the trouble to set .flags.writeable=False?

    @classmethod
    def new(cls, xk, c):
        """Make a new PwPoly with given knots and coefficients.

        Parameters
        ----------
        xk : ndarray, shape (K) or (K+1)
        c : ndarray, shape (N, ..., K+1)

        Returns
        -------
        p : PwPoly or a subclass
            The new PwPoly.

        Notes
        -----
        If ``xk.size`` is K+1, `xk` will not be copied.  If
        ``xk.size`` is K, ``xk[0]`` will be duplicated.  The
        coefficient array `c` is always used uncopied.  No error
        checking is done; this is a low level method for creating
        PwPoly results, not a user interface.
        """
        pwp = cls()
        cls._rawinit(pwp, xk, c)
        return pwp

    @classmethod
    def _two_sided_inverse(cls, h):
        # worker for __init__:  Compute and cache inverse of binom(m,k)
        # where 0<=k<=h and h+1<=m<=h, needed to find the coefficients
        # of x**m for an expansion on the left side of an interval, given
        # the coefficients of (x-1)**k on the right side.
        m = cls._two_sided_cache.get(h)
        if m is None:
            # not particularly efficient, but terse
            m = polyddx(eye(2*h+2, h+1, -h-1), 1., h)
            cls._two_sided_cache[h] = m = inv(m)
        # Note that the matrix becomes ill-conditioned quickly as h increases
        # should be fine for polynomials up to degree 10 or so (h~5).  The
        # whole strategy of the PwPoly algorithm for computing the polynomials
        # in the first place will not work for high degree polynomials.
        #   m.dot(rhs) = lhs, sum[j]){m_ij * rhs_j} = lhs_i
        # where rhs are coefficients 0...h and lhs are coefficients h+1...2h+1
        return m
    _two_sided_cache = {}

    def __call__(self, x, nd=0):
        """Compute piecewise polynomial function at points x.

        Parameters
        ----------
        x : array_like
            Points to evaluate function.
        nd : optional int, default 0
            Number of derivatives to evaluate.

        Returns
        -------
        y : ndarray, or tuple of ``1+nd`` ndarrays when ``nd>0``
            Function values, or function and derivative values.  Each array
            has same same shape as `x`, unless the piecewise function was
            defined with additional dimensions, in which case those become
            the leading dimensions of the result arrays.
        """
        x = asfarray(array(x))    # make copy here for -= below
        ix = searchsorted(self.xk, x)
        x -= self.xk0[ix]
        c = self.c[..., ix]
        return polyddx(c, x, nd, True) if nd else polyfun(c, x)

    def degree(self):
        """Highest power of the piecewise polynomial as a method."""
        # method rather than property to match numpy Polynomial class
        return self.c.shape[0] - 1

    @property
    def deg(self):
        """Highest power of the piecewise polynomial as a property."""
        return self.c.shape[0] - 1

    @property
    def ndim(self):
        """Number of dimensions of result when evaluated at a scalar x."""
        return self.c.ndim - 2

    @property
    def shape(self):
        """Shape of result when evaluated at a scalar x."""
        return self.c.shape[1:-1]  # first dimension is degree, last is knot

    def deriv(self, m=1):
        """Derivative as a new piecewise polynomial.

        Parameters
        ----------
        m : int
            The number of derivatives to perform.

        Returns
        -------
        PwPoly
            The derivative of the PwPoly.
        """
        c = self.c
        s = (1,)*(c.ndim - 1)
        for _ in range(m):
            n = c.shape[0]
            if n < 2:
                c = zeros_like(c)
                break
            c = c[1:] * arange(1, n).reshape((n-1,) + s)
        return self.new(self.xk0, c)

    def integ(self, m=1, k=None, lbnd=0):
        """Integral as a new piecewise polynomial.

        Parameters
        ----------
        m : int
            The number of integrations to perform.
        k : array_like
            Integration constants. The first constant is applied to the
            first integration, the second to the second, and so on. The
            list of values must less than or equal to `m` in length and any
            missing values are set to zero.
        lbnd : Scalar
            The lower bound of the definite integral.

        Returns
        -------
        PwPoly
            The integral of the PwPoly.
        """
        c = self.c
        consts = zeros((m,) + c.shape[1:-1])
        lbnd = asfarray(lbnd)
        if lbnd.ndim:
            raise TypeError("lower bound must be scalar")
        dxlb = self.xk0 - lbnd
        if k is not None:
            k = asfarray(k)
            if k.ndim == c.ndim-2:
                k = k[newaxis]
            if k.shape[0] < m:
                k = concatenate((k, zeros((m-k.shape[0],)+k.shape[1:])))
            consts += k
        # integration constants same for every interval xk
        consts = consts[..., newaxis] + zeros(c.shape[-1])
        for k in consts[:, newaxis]:  # add leading (1,) axis for concatenate
            n = c.shape[0]
            n = 1. / arange(1, n+1).reshape((n,) + (1,)*(c.ndim-1))
            c = concatenate((k, c*n))
            # c[0] each have lower bound at xk, adjust to common lbnd
            c[0] -= polyfun(c, dxlb)
        return self.new(self.xk0, c)

    def roots(self, value=0., tol=1.e-9):
        """Return the values of x for which the piecewise polynomial is zero.

        Only works for scalar PwPoly.

        Parameters
        ----------
        value : float, optional
            Return values of `x` for which piecewise polynomial equals value.
            Defaults to 0.0.

        Returns
        -------
        ndarray
            1D array of `x` values where this PwPoly evaluates to `value`.
        """
        # ideas:
        # (1) check jumps for discontinuous roots
        # (2) Use Descartes rule of signs to eliminate intervals with
        #     no possibility of roots.  First, count count sign changes in
        #     coefficients as given -- if none, eliminate this interval.
        #     Next, compute coefficients at end of interval, change
        #     sign of odd coefficients, and count sign changes.  If none,
        #     eliminate the interval.  Side effect: eliminates constant
        #     intervals.
        # (3) special case piecewise linear
        # (4) use eigenvalue solver for degree>=2
        c = self.c.copy()
        if c.ndim != 2:
            raise TypeError("cannot find roots of multidimensional function")
        if value:
            c[0] -= value
        xk, xk0 = self.xk, self.xk0
        dx = xk - xk0[:-1]
        cup = polyddx(c[:, :-1], dx)  # coefficients at end of intervals
        yhi = cup[0]
        ylo = c[0, 1:]
        x = xk[yhi*ylo <= 0.]  # list of knots where jumps across zero
        n = c.shape[0] - 1
        if n < 1:
            return x
        if len(dx) > 1:
            dx[0] = dx[1]
            dx = concatenate((dx, dx[-1:]))
        else:
            dx = array([1., 1.])
        dxsave = dx
        dxn = dx + zeros((n+1,)+dx.shape)
        dxn[0] = 1
        dxn = dxn.cumprod(axis=0)  # [1, dx, dx**2, ..., dx**n]
        c *= dxn
        cup *= dxn[:, :-1]  # scale each interval to (0,1)
        # Descartes rule of signs: must be at least one sign change if root>0
        maybe = ((c[1:] < 0) != (c[:-1] < 0)).sum(axis=0) > 0
        maybe[0] = True  # no dx>0 test for left semi-infinite interval
        cup[1::2] = -cup[1::2]  # change sign of odd upper coefficients
        maybe[:-1] &= ((cup[1:] < 0) != (cup[:-1] < 0)).sum(axis=0) > 0
        c, dx, xk0 = c[:, maybe], dx[maybe], xk0[maybe]
        deg = absolute(c)
        deg = (deg > tol*deg.max(axis=0)) * arange(n+1)[:, newaxis]
        deg = deg.max(axis=0)  # effective degree in each interval
        x = [x] if x.size else []
        linear = (deg == 1)
        if any(linear):
            cup = c[0:2, linear]
            xi = -cup[0, :] / cup[1, :]
            mask = (xi >= 0.)
            # semi-infinite intervals are special cases
            if maybe[0] and linear[0]:
                mask[0] = (xi[0] <= 0.)
            elif not (maybe[-1] and linear[-1]):
                mask &= (xi <= 1.)
            if any(mask):
                xi = xi[mask]
                linear[linear] &= mask
                x.append(xi*dx[linear] + xk0[linear])
        linear = (deg > 1)  # actually means non-linear here
        check_first = maybe[0] and linear[0]
        check_last = maybe[-1] and linear[-1]
        # for non-linear intervals, fall back to generic root finder
        # this loop over intervals is slow but unavoidable
        c, dx, xk0 = c[:, linear], dx[linear], xk0[linear]
        nold, imax = -1, len(xk)
        for i, ci in enumerate(c.T):  # loop on knot intervals
            n = deg[i]
            if n != nold:
                m = zeros((n, n), dtype=c.dtype)
                m.reshape(n*n)[n::n+1] = 1
            m[:, -1] = -ci[:n] / ci[n]
            xi = eigvals(m)
            xok, xi = isreal(xi), real(xi)
            if (i == 0) and check_first:
                xok &= (xi <= 0.)
            elif (i < imax) or not check_last:
                xok &= (xi >= 0.) & (xi <= 1.)
            else:
                xok &= (xi >= 0.)
            if not any(xok):
                continue
            x.append(xi[xok]*dx[i] + xk0[i])  # undo (0,1) dx c scaling
        x = concatenate(x)
        x.sort()
        # remove duplicates or near duplicates
        if len(x) > 1:
            mask = ones(x.shape, dtype=bool)
            i = xk.searchsorted(x)
            dx = dxsave[i]
            d = x[1:] - x[:-1]
            dx = 0.5*(dx[1:] + dx[:-1])
            mask[1:] = d > tol*dx
            x = x[mask]
        return x

    def jumps(self, n=None):
        """Return jumps in polynomial coefficients at the knot points.

        Parameters
        ----------
        n : int, optional
            Maximum degree of returned coefficients, all by default.

        Returns
        -------
        dc : ndarray
            Shape is ``(n, ..., K)`` where `K` is number of knot points.
            Degree is first axis, ``dc[0]`` is jump in function value,
            ``dc[1]`` is jump in derivative, ``dc[2]`` is **half** the jump
            in second derivative, and so on.  Knot is final axis.  Any
            additional axes are the dimensionality of the PwPoly.
        """
        if n is None:
            n = self.c.shape[0] - 1
        dx = self.xk0[1:] - self.xk0[:-1]
        cminus = polyddx(self.c[..., :-1], dx, n)  # value just below xk
        return self.c[0:1+n, ..., 1:] - cminus

    def reknot(self, xk, n=None):
        """Return a new PwPoly with knot points xk that approximates this one.

        If `n` is the polynomial degree, the resulting piecewise
        polynomial will match ``(n-1)/2`` derivatives of the old one
        at the points `xk`.  If `n` is even, the n-th derivative at
        the midpoint of each `xk` interval will also match the
        original.

        Parameters
        ----------
        xk : array_like
            1D array of new knot points.
        n : int, optional
            Degree for new piecewise polynomial, default same as this one.

        Returns
        -------
        PwPoly
            Like this PwPoly but with knots `xk`.
        """
        if n is None:
            n = self.c.shape[0] - 1
        xk = asarray(xk)
        if xk.ndim != 1:
            raise TypeError("new knot points must be 1D array_like")
        if xk[0] > xk[-1]:
            xk = xk[::-1]  # permit decreasing xk as convenience
        isper = isinstance(self, PerPwPoly)
        if isper and (absolute(xk[-1]-xk[0]-self.period) > 1.e-6*self.period):
            raise ValueError("new PerPwPoly knot"
                             "must have same period as old")
        h = maximum((n - 1)//2, 0)
        c = self(xk, h)
        if not h:
            c = c[newaxis]
        c = tuple(c)
        if not (n & 1):  # breaks if only one knot point...
            c += (self.deriv(n)(0.5*(xk[1:] + xk[:-1])),)
        pw = object.__new__(self.__class__)
        PwPoly.__init__(pw, xk, *c)
        if isper:
            pw.period = xk[-1] - xk[0]
        return pw

    def addknots(self, xk, _nocheck=False):
        """Return a new PwPoly with knot points xk that matches this one.

        The `xk` must be a superset of the existing knot points for this to
        work as intended.

        Parameters
        ----------
        xk : array_like
            1D array of new knot points.
        n : int, optional
            Degree for new piecewise polynomial, default same as this one.

        Returns
        -------
        PwPoly
            Like this PwPoly but with knots `xk`.
        """
        if not _nocheck:
            xk = self.allknots(xk)
            if xk is self.xk:
                return self
            if xk.ndim != 1:
                raise TypeError("new knot points must be 1D array_like")
            elif len(xk) == len(self.xk):
                return self
        ic = self.xk.searchsorted(0.5 * (xk[1:] + xk[:-1]))
        ic = concatenate(([0], ic, [len(self.xk)]))
        xk0 = concatenate((xk[0:1], xk))
        # ic is index of interval in self.xk for each interval bounded by xk
        c = polyddx(self.c[..., ic], xk0-self.xk0[ic])
        return self.new(xk, c)

    def allknots(self, *args, **kwargs):
        """Return union of knot points among piecewise polynomials.

        Parameters
        ----------
        pwp1,pwp2,... : PwPoly
            The input PwPoly instances.
        tol : float, optional keyword
            Relative tolerance compared to interval between knots,
            default 1e-6.

        Returns
        -------
        xk : ndarray
            1D union of knot points in increasing order.
        """
        tol = kwargs.pop('tol', 1.e-6)
        if kwargs:
            raise TypeError("unrecognized keyword argument")
        xk = self.xk
        for a in args:
            try:
                yk = a.xk
            except AttributeError:
                yk = a
            if array_equal(xk, yk):
                continue
            if len(yk) > len(xk):
                xk, yk = yk, xk
            iy = searchsorted(xk, yk)
            dx = xk[1:] - xk[:-1]
            if len(dx):
                dx = concatenate((dx[0:1], dx, dx[-1:]))[iy] * tol
                i, j = maximum(iy-1, 0), minimum(iy, len(xk)-1)
                mask = absolute(yk - xk[i]) > dx
                mask &= absolute(yk - xk[j]) > dx
                yk = yk[mask]
            elif isclose(xk[0], yk[0], rtol=tol, atol=tol):
                continue
            xk = sort(concatenate((xk, yk)))
        return xk

    # pickle and copy
    def __getstate__(self):
        return dict(xk0=self.xk0.copy(), c=self.c.copy())

    def __setstate__(self, d):
        self.xk0 = d['xk0']
        self.c = d['c']
        self.xk = self.xk0[1:]

    def __repr__(self):
        xk, c = repr(self.xk)[6:-1], repr(self.c)[6:-1]
        name = self.__class__.__name__
        return "{}({}, {})".format(name, xk, c)

    def __str__(self):
        nk, deg, s = self.xk.size, self.deg, self.shape
        s = ", shape="+str(s) if s else ""
        name = self.__class__.__name__
        return "{}({} knots, degree {}{})".format(name, nk, deg, s)

    def __len__(self):
        s = self.shape
        if not s:
            raise TypeError("scalar PwPoly has no len()")
        return s[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        colon = slice(None)
        return self.new(self.xk0, self.c[(colon,)+key+(Ellipsis, colon)])

    def __neg__(self):
        return self.new(self.xk0, -self.c)

    def __pos__(self):
        return self  # no point in a copy here?

    # binary operations distinguish
    # (1) other operand an ndarray
    # (2) other a PwPoly, but different degree and/or knot points
    # (3) other a PwPoly, but different additional dimensions
    def _binary(self, other, skip=False):
        this = self
        if isinstance(other, PwPoly):
            if not skip:
                nd = maximum(this.deg, other.deg)
                if this.deg != nd:
                    z = list(this.c.shape)
                    z[0] = nd - this.deg
                    z = zeros(z, dtype=this.c.dtype)
                    this = self.new(this.xk0, concatenate((this.c, z)))
                if other.deg != nd:
                    z = list(other.c.shape)
                    z[0] = nd - other.deg
                    z = zeros(z, dtype=other.c.dtype)
                    other = self.new(other.xk0, concatenate((other.c, z)))
            sa, sb = this.c.shape, other.c.shape
            la, lb = len(sa), len(sb)
            if la < lb:
                this = self.new(this.xk0,
                                this.c.reshape(sa[0:1]+(1,)*(lb-la)+sa[1:]))
            elif la > lb:
                other = self.new(other.xk0,
                                 other.c.reshape(sb[0:1]+(1,)*(la-lb)+sb[1:]))
            xk = this.allknots(other)
            if xk is not this.xk:
                this = this.addknots(xk, True)
                other = other.addknots(xk, True)
            other = other.c
            numb = False
        else:
            other = asfarray(other, dtype=self.c.dtype)[..., newaxis]
            numb = True
        return this, other, numb

    def __add__(self, other):
        this, other, numb = self._binary(other)
        return self.new(this.xk0, this.c+other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        this, other, dummy = self._binary(other)
        return self.new(this.xk0, this.c-other)

    def __rsub__(self, other):
        this, other, dummy = self._binary(other)
        return self.new(this.xk0, other-this.c)

    def __mul__(self, other):
        this, other, numb = self._binary(other, True)
        c = this.c
        if numb:
            return self.new(this.xk0, c*other)
        # multiply polynomials
        n, m = c.shape[0], other.shape[0]
        if n < other.shape[0]:
            c, other, n, m = other, c, m, n
        product = None
        for i, d in enumerate(other):
            cd = c * d
            if product is None:
                product = zeros((n+m-1,)+cd.shape[1:])
            product[i:i+n] += cd
        return self.new(this.xk0, product)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        this, other, numb = self._binary(other)
        if not numb:
            raise TypeError("cannot divide by a PwPoly")
        return self.new(this.xk0, this.c / other)


class PerPwPoly(PwPoly):
    """Periodic piecewise polynomial function.

    Like PwPoly, except first and last knot points `xk` are the same point,
    but advanced by one period.  Input arguments to the function will be
    reduced to this period before evaluation, so there is no extrapolation.

    Note that when you initialize a PerPwPoly, you must explicitly
    supply the duplicate final point at the end of the period.
    """
    def __call__(self, x, nd=0):
        x0 = self.xk[0]
        x = x0 + (x - x0)%self.period
        return super(PerPwPoly, self).__call__(x, nd)

    def _rawinit(self, xk, c):
        super(PerPwPoly, self)._rawinit(xk, c)
        if self.xk.size < 2:
            raise TypeError("cannot create periodic function with <2 knots")
        # set extrapolation to guard against roundoff error
        self.c[..., 0] = self.c[..., 1]  # potentially modify input c array...
        self.c[..., -1] = polyddx(self.c[..., -2], self.xk[-1] - self.xk[-2])
        self.period = self.xk[-1] - self.xk[0]

########################################################################
# alternative PwPoly constructors are implemented as functions


def spline(x, y, n=3, lo=(), hi=(), per=False, extrap=None):
    """Construct a PwPoly as a spline through given points (x,y).

    A spline is a piecewise polynomial of degree n (default 3) passing
    through all of the given points with continuous derivatives up to
    degree n-1 at every point.  Before ``x[0]`` and after ``x[-1]``,
    the result will be of degree n-1, maintaining continuity of n-1
    derivatives at every knot point, including the first and last.

    Parameters
    ----------
    x : array_like
        1D array of knot points, strictly monotonic.
    y : array_like
        The given ``(x,y)`` data points to be fit.  The `y` data may
        have additional leading dimensions, in order to fit a curve in
        a multi-dimensional space.
    n : int, optional
        The degree of the PwPoly.  Defaults to 3, a piecewise cubic spline.

    lo, hi : tuple of array_like, optional
        Boundary conditions at the endpoints ``x[0]`` and ``x[-1]``.
        Each tuple represents the values of ``(dydx, d2ydx2/2,
        d3ydx3/6, ...)`` at ``x[0]`` for `lo` or ``x[-1]`` for `hi`.
        Use None for a derivative to be not specified.  The
        derivatives are broadcast to any leading dimensions of `y` if
        necessary.

        The highest derivative you can specify is n-1, that is
        `d2ydx2/2` for a cubic spline, `dydx` for a quadratic spline,
        and so on.  For the special case that you only wish to specify
        `dydx` at an endpoint, `lo` or `hi` need not be a tuple
        ``(dydx,)``; spline accepts simply `dydx` to mean ``(dydx,
        None, None, ...)``.  The maximum number of derivatives you can
        specify in both `lo` and `hi` is n-1, that is, 2 for a cubic
        spline, 4 for a pentic, and so on.  If you specify fewer than
        n-1 derivatives, spline will set the highest unspecified
        derivatives to zero, begining with `lo` and alternating `lo`,
        `hi`, `lo`, etc., until n-1 derivative are specified.  For
        example, if you specify neither `lo` nor `hi` for a cubic
        spline, then spline sets ``d2ydx2=0`` at both endpoints, and
        the natural spline with zero curvature at the endpoints is the
        result.  This is equivalent to ``lo=(None,0.)``, ``hi=(None,0.)``.

    per : bool
        True to make all derivatives match at first and last points of x.
        This produces a PerPwPoly function, which maps any inputs
        outside the first and last points of x into that interval.

    extrap : int or (int, int)
        Degree to maintain continuity beyond endpoints, can be set separately
        for before first and after last interval.

    See Also
    --------
    pline : polyline constructor, use instead of spline for linear case
    splfit : construct a pline by fitting points rather than interpolating


    Notes
    -----
    The K knots define K-1 intervals, requiring a total of
    ``(n+1)*(K-1)=n*K-n+K-1`` polynomial coefficients.  There are
    ``n*(K-2)`` continuity constraints at the interior knots and K `y`
    values, for a total of ``n*K-2*n+K`` constraint equations.  We
    require n-1 additional constraints to determine a unique spline
    fit.  When n is odd (linear, cubic, pentic, etc.), these
    additional constraints may be divided equally between the first
    and last knots points.  When n is even, there is no way to treat
    the first and last knots symmetrically.

    By default, spline forces the ``(n-1)//2`` highest derivatives to
    zero at each endpoint, with the first endpoint getting an extra
    zero derivative in the case of even n.  The lo and hi keywords
    allow you to split up the n-1 constraints among the possible
    derivatives and endpoints however you wish.
    """
    # any way to generate other members of family with different BCs?
    # yes -- just pass 0*y and desired BCs
    x, y = asfarray(x), asfarray(y)
    dtype = promote_types(x.dtype, y.dtype)
    x, y = x.astype(dtype), y.astype(dtype)
    nk = x.size
    if x.ndim != 1 or nk < 2:
        raise TypeError("x must be 1D array_like with at least two points")
    shape = y.shape
    if shape[-1] != nk:
        raise TypeError("y must have same final axis as x")
    shape = shape[:-1]
    nshape = prod(shape)
    y = y.reshape(nshape, nk)
    nm1, nk1 = n-1, nk-1
    if not isinstance(lo, tuple):
        lo = (lo,)
    if not isinstance(hi, tuple):
        hi = (hi,)
    if per and (lo or hi or (extrap is not None)):
        raise TypeError("periodic bcs preclude lo, hi, or extrap")
    if extrap is None:
        extrap = (nm1, nm1)
    elif not isinstance(extrap, tuple):
        extrap = (extrap, extrap)
    if x[0] > x[-1]:  # as a convenience, permit monotonic decreasing x
        x, y = x[::-1].copy(), y[..., ::-1].copy()
        lo, hi = hi, lo
        extrap = extrap[::-1]
    if len(lo) > nm1 or len(hi) > nm1:
        raise TypeError("cannot specify more than n-1st derivative in bc")
    if per:
        y = y.copy()
        y[:, -1] = y[:, 0]  # ignore given y[-1], assume same as y[0]
    if n < 2:
        y = y.reshape(shape+(nk,))
        p = PerPwPoly(x, y) if per else PwPoly(x, y)
        xk, c = p.xk, p.c
        c[..., 0] = c[..., 1]
        c[extrap[0]+1:, ..., 0] = 0
        if len(xk) > 1:
            c[..., -1] = polyddx(p.c[..., -2], xk[-1] - xk[-2])
            c[extrap[1]+1:, ..., -1] = 0
        return p
    lo += (None,)*(nm1 - len(lo))
    lo = lo[:nm1]  # lo must always have exactly nm1 items for later
    nlo = nm1 - lo.count(None)
    nhi = len(hi) - hi.count(None)
    missing = nm1 - (nlo+nhi)
    if missing and not per:
        # insert default boundary conditions where unspecified
        lo = list(lo)
        hi = list(hi + (None,)*(nm1 - len(hi)))
        i = n - 2
        while missing:
            if i < 0:
                raise IndexError("impossible error, logic bug?")
            if lo[i] is None:
                lo[i] = 0.    # fill in missing BC with zero
                missing -= 1
                nlo += 1
            if missing and hi[i] is None:
                hi[i] = 0.    # fill in missing BC with zero
                missing -= 1
                nhi += 1
            i -= 1
        lo = tuple(lo)
        hi = tuple(hi)
    bc = _bico(n, dtype=dtype)[:, 1:]
    # [[  1.,   1.,   1.,   1.,   1.],
    #  [  1.,   2.,   3.,   4.,   5.],
    #  [  0.,   1.,   3.,   6.,  10.],
    #  [  0.,   0.,   1.,   4.,  10.],
    #  [  0.,   0.,   0.,   1.,   5.]]  for example, when n=5
    bc[0, -1] = 0  # subdiagonal, no continuity equation for n-th derivative
    ab = bc[::-1]  # solve_banded diagonal form, upper diagonals first
    nun = nk1 * n  # (nk-1)*n unknowns and equations
    ab = ab[:, newaxis].repeat(nk-1, axis=1).reshape(n, nun)
    dx = x[1:] - x[:-1]
    dxr = dx / roll(dx, -1)  # current/next interval width, periodic case
    dxr = dxr[:, newaxis].repeat(n, axis=1)
    dxr[:, 0] = -1
    dxr = dxr.cumprod(axis=1)
    dxr[:, 0] = 1  # 1, -R, -R**2, -R**3, -R**4, 1, -R, ...
    ab = concatenate((roll(dxr.ravel(), nm1)[newaxis], ab))
    b = zeros((nun, nshape), dtype)
    y = y.T
    b[0::n] = y[1:] - y[:-1]
    # compute powers of dx for normalization, [dx, dx**2, ... dx**n]
    dxr = dx[:, newaxis].repeat(n, axis=1).cumprod(axis=1)
    if per:
        b = solve_periodic((1, nm1), ab, b, overwrite_ab=True,
                           overwrite_b=True, check_finite=False)
    else:
        ab[0, :nm1] = 0  # unused periodic elements
        dxb, bb, bc = dxr[-1], b[-nm1:], bc[1:]
        k = -1
        for i, bnd in enumerate(hi):
            if bnd is not None:
                k += 1
                bb[k] = bnd * dxb[i]  # normalize value with dx**p
                kmn = k - n
                ab.ravel()[kmn:kmn*nun:1-nun] = bc[i, k:]
        if nlo:
            # lo BCs shift equations, number of lower and upper diagonals
            b = roll(b, nlo, axis=0)
        dxb = dxr[0]
        j, k = nm1*nun, nlo-1
        for i, bnd in enumerate(reversed(lo)):
            if bnd is not None:
                mi = nm1-1 - i
                b[k] = bnd * dxb[mi]  # normalize value with dx**p
                k -= 1
                j -= nun
                abeq = ab.ravel()[j::1-nun]
                abeq[mi] = 1
        b = solve_banded((1+nlo, nm1-nlo), ab, b, overwrite_ab=True,
                         overwrite_b=True, check_finite=False)
    rdxn = 1./dxr.ravel()
    b *= rdxn[:, newaxis]
    b = concatenate((y[:-1, newaxis], b.reshape(nk-1, n, nshape)), axis=1)
    b = transpose(b, (1, 2, 0)).copy()
    b = b.reshape((n+1,) + shape + (nk-1,))
    c = polyddx(b[..., -1:], dx[-1:])
    c = concatenate((b[..., 0:1], b, c), axis=-1)
    if per:
        return PerPwPoly.new(x, c)
    else:
        c[extrap[0]+1:, ..., 0] = 0
        c[extrap[1]+1:, ..., -1] = 0
        return PwPoly.new(x, c)


def pline(x, y, extrap=None, per=False):
    """Return a piecewise linear function through given points (x,y).

    Convenience shorthand for ``spline(x, y, n=1)``, intended as a more
    general version of the interp function with a slightly different
    interface::

        pline(x, y)(xp)  <--->  interp(xp, x, y)

    But pline permits `y` to have additional dimensions and `x` can be
    a decreasing series.

    Note that ``pline(x, y)`` by default is constant before the first
    `x` and after the final `x`.  Use ``pline(x, y, 1)`` to
    extrapolate using the linear function in the first and last
    intervals.

    See Also
    --------
    spline : general PwPoly spline interpolation function
    """
    return spline(x, y, n=1, extrap=extrap, per=per)


def _plfitter(adiag, asup, b, lo=None, hi=None, per=False, **kwargs):
    y = b.copy()
    if per:
        a = array([adiag[:-1], asup[:-1]])  # symmetric tridiag, lower form
        a[0, 0] += adiag[-1]
        b[0] += b[-1]
        y[:-1] = solves_periodic(a, b[:-1], lower=True,
                                 check_finite=False, overwrite_b=True)
        y[-1] = y[0]
    else:
        a = array([adiag, asup])  # symmetric tridiag, lower form
        if lo is not None:
            y[0] = lo
            a = a[:, 1:]
            b = b[1:]
            b[0] -= lo * asup[0]
            lo = 1
        if hi is not None:
            y[-1] = hi
            a = a[:, :-1]
            b = b[:-1]
            b[-1] -= hi * asup[-2]  # asup[-1] is zero padding
            hi = -1
        # Note that solveh_banded only handles positive definite a.
        # Without lo or hi, a is positive definite, because y.T*a*y = chi2.
        # With lo or hi constraints, chi2 minus first or last term must
        # still be positive?  Not obvious...
        y[lo:hi] = solveh_banded(a, b, lower=True,
                                 check_finite=False, overwrite_b=True)
    return y


def plfit(xk, x, y, sigy=None, lo=(), hi=(), per=False, extrap=None,
          solver=_plfitter, **kwargs):
    """Return best fit linear PwPoly with knots xk to given points (x,y).

    Gives the same fit as ``splfit(xk, x, y, n=1, nc=0)``, that is, the
    best fit piecewise polynomial through ``(x,y)`` with knots `xk`.
    See splfit documentation for detailed description of the common
    parameters.  If you need the cost of continuity constraints, use
    splfit instead.

    See Also
    --------
    splfit : general PwPoly spline fitting function
    """
    # Implement using order 2 B-spline algorithm, a completely different
    # strategy from the splfit algorithm.
    xkorig, xk, x, y, lo, hi, extrap = _splfit_setup(xk, x, y, lo, hi,
                                                     per, extrap)
    if len(lo) > 1 or len(hi) > 1:
        raise ValueError("BCs on derivatives in lo, hi not supported in pline")
    x, y, sigy, ix, dxk, yshape, lo, hi = _splfit_args(xk, x, y, sigy, lo, hi)
    lo = lo[0] if lo else None
    hi = hi[0] if hi else None
    nun = xk.size       # number of unknowns
    one = array(1., dtype=y.dtype)
    w = one / (sigy * sigy)  # chi2 weights
    rdxk = one / dxk
    p = (x - xk[ix]) * rdxk[ix]
    q = one - p
    yk = []
    ll, hh = None, None
    # This needs to be a loop to allow for the possibility that sigy
    # may be different for each component of y, which is the only
    # dependence of the matrix to be solved on y component.
    for i, yy in enumerate(y):
        wp = w[i]
        wp, wq = wp*p, wp*q
        adiag = bincount(ix, wq*q, nun) + bincount(1+ix, wp*p, nun)
        asup = bincount(ix, wp*q, nun)
        b = bincount(ix, wq*yy, nun) + bincount(1+ix, wp*yy, nun)
        if lo is not None:
            ll = lo[i]
        if hi is not None:
            hh = hi[i]
        yk.append(solver(adiag, asup, b, lo=ll, hi=hh, per=per, **kwargs))
    yk = array(yk).reshape(yshape+(nun,))  # put back actual shape of yk
    return pline(xk, yk, extrap=extrap, per=per)


def splfit(xk, x, y, sigy=None, n=3, nc=None, lo=(), hi=(), per=False,
           extrap=None, cost=None):
    """Return the best fit PwPoly with knots xk to given points (x,y).

    Parameters
    ----------
    xk : 1D array_like
        Knot points of the PwPoly, sparse enough that there are some
        `x` values "near" each interval of `xk`.  The `xk` must be
        strictly monotonic.

    x,y : array_like
        Cloud of points to fit.  If `f` is the returned PwPoly,

            ``chi2 = sum(((f(x) - y)/sigy)**2)``

        will be minimum for the given PwPoly knot points `xk`, degree n,
        and boundary constraints.  The `y` array must have the same or
        conformable trailing dimensions as `x`.  However, `y` may have
        additional leading dimensions to produce a multi-dimensional
        PwPoly curve.  In this case, the `chi2` sum extends over all
        the leading dimensions for each point.  The order of the ``(x,y)``
        points is irrelevant; indeed `x` need not be a 1D array.

    sigy : array_like, optional
        Standard deviations of `y`, must be conformable to `y`, default 1.

    n : int, optional
        The degree of the PwPoly, default 3 (piecewise cubic).

    nc : int, optional
        The degree to which the result PwPoly is continuous, default
        n-1 (that is, the PwPoly is a spline of degree n).  Note that
        a smaller value of nc requires more ``(x,y)`` points per `xk`
        interval.

    lo, hi : tuple of array_like, optional
        Boundary conditions at the endpoints ``xk[0]`` and ``xk[-1]``.
        Each tuple represents the values of ``(y, dydx, d2ydx2/2,
        d3ydx3/6, ...)`` at ``xk[0]`` for `lo` or ``xk[-1]`` for `hi`.
        Use None for a derivative to be not specified.  See the
        docstring for spline for details; however, note that the `lo`
        and `hi` tuples for splfit begin with the function value
        (zero-th derivative), while the `lo` and `hi` tuples for
        spline begin with the first derivative.

    per : bool
        True to make all derivatives match at first and last points of
        `xk`.  This produces a PerPwPoly function, which maps any
        inputs outside the first and last points of `xk` into that
        interval.

    extrap : int or (int,int)
        Degree to maintain continuity beyond endpoints, can be set separately
        for before first and after last interval.

    cost : bool, optional
        Set True to return cost array continuity of various degrees.

    Returns
    -------
    fit : PwPoly
        The best fit to the given ``(x,y)`` with the given knots `xk`.
    cost : tuple of ndarray
        This return is only present with ``cost=1`` parameter.  The
        Lagrange multipliers used to enforce the `chi2` minimization
        constraints associated with the continuous function values and
        nc derivative values, plus any additional boundary condition
        constraints you specified with the `lo` or `hi` keywords.  The
        interior knot point constraints are ``cost[0]`` for function
        continuity, ``cost[1]`` for first derivative continuity,
        through ``cost[nc]`` for nc-derivative continuity.  Then
        ``cost[nc+1:nc+1+nlo]`` are the costs of the `lo` constraints
        and ``cost[nc+1+nlo:]`` are the costs of the `hi` constraints,
        where ``nlo=len(lo)``.

        The shape of each of the ``cost[:nc+1]`` interior constraints
        is ``y[...,1:-1].shape``, that is, any leading dimensions of
        `y` followed by the number of interior knot points where the
        constraints apply.  The shape of each of the ``cost[nc+1:]``
        boundary constraints is ``y[...,0].shape``, that is, any
        leading dimensions of `y`.

    See Also
    --------
    plfit : special case of splfit for ``n=1``, ``nc=0``
    spline : general PwPoly spline interpolation function

    Notes
    -----
    The cost values require a bit more explanation.  Each interior
    constraint is schematically ``c[this] - l2r.dot(c[prev]) = 0``
    where `l2r` is the linear transformation mapping the polynomial
    coefficients c[prev] in the interval to the left to their values
    at ``xk[this]`` knot.  That is, each interior constraint is that
    the jump in the value of the polynomial coefficient of some degree
    be zero.  The cost is defined to be the partial derivative of
    `chi2` with respect to that jump.  In other words, if we relaxed
    just that one constraint and allowed a jump of `dc`, `chi2` would
    change by ``cost*dc``.  We keep this sign convention for the `lo`
    and `hi` constraints as well -- a positive `cost` means that if
    the coefficient changes by a positive amount for increasing x
    across the knot, then `chi2` will increase.
    """
    # There is a degree-n B-spline algorithm analogous to the degree-1
    # algorithm used in plfit.  The direct matrix solve here is far
    # less messy, though more equations for the matrix solve.  The
    # additional unknowns are the Lagrange multipliers, which
    # represent the cost of the continuity constraints.  The conversion
    # from B-spline control points back to PwPoly coefficients is also
    # non-trivial.
    xkorig, xk, x, y, lo, hi, extrap = _splfit_setup(xk, x, y, lo, hi,
                                                     per, extrap)
    x, y, sigy, ix, dxk, yshape, lo, hi = _splfit_args(xk, x, y, sigy, lo, hi)
    if nc is None:
        nc = n - 1  # degree of continuity, nc+1 constraints
    elif nc < 0 or nc >= n:
        raise ValueError("illegal nc, bigger than n-1 or less than 0")
    if maximum(len(lo), len(hi)) > 1+nc:
        raise ValueError("hi or lo longer than degree of continuity nc")
    if not per:
        if extrap is None:
            extrap = nc
        extrap = asarray(extrap) + zeros(2, dtype=int)
    dtype = y.dtype
    nk1 = dxk.size  # number of intervals between knots
    one = array(1., dtype=dtype)
    w = one / (sigy * sigy)  # chi2 weights
    rdxk = one / dxk
    x = (x - xk[ix])*rdxk[ix]
    n1, n2, nc1 = 1+n, 2+n, 1+nc
    dxscl = dxk.reshape(nk1, 1).repeat(n1, axis=1)
    dxscl[:, 0] = 1
    dxscl = dxscl.cumprod(axis=1)
    rat = dxscl[:, :nc1] / roll(dxscl[:, :nc1], -1, axis=0)
    ash0 = (n2, nk1, n2+nc)   # initial shape of lower diagonal form
    ash1 = (n2, (n2+nc)*nk1)  # final shape, before lo/hi constraints
    nhi, nlo = len(hi), len(lo)
    nstrip = nc1 - nhi
    if lo:
        alo = zeros((n2, nlo))
        alo[nlo, :] = -1  # nlo additional Lagrange multipliers (costs)
        zblo = zeros(nlo)
    # Need to loop here because matrix a potentially different for
    # each component of y due to differing sigy.
    cco = _bico(n, nc).ravel()  # coefficients for constraint equations
    c = []
    for i, wi in enumerate(w):
        yi = y[i]
        wx = wi.copy()
        b = zeros((nk1, n2+nc))  # always dtype float (double precision)
        xp = zeros((nk1, n1+n))
        for j in range(n1):
            b[:, j] = bincount(ix, wx*yi, nk1)  # sum(w*y*x**p)
            xp[:, j] = bincount(ix, wx, nk1)
            wx *= x
        for j in range(1, n1):
            xp[:, n+j] = bincount(ix, wx, nk1)  # sum(w*x**p)
            if j < n:
                wx *= x
        # build matrix a in symmetric lower diagonal form
        a = zeros(ash0)
        for j in range(n1):
            a[j, :, 0:n1-j] = xp[:, j:n1+n-j:2]
            if j:
                cc = cco[n1-j:n1*j:n2]
                a[j, :, n1-j:n1-j+cc.size] = cc
        a[n1, :, :nc1] = 1
        a[nc1, :, -nc1:] = -rat[:, 0:nc1]
        a = a.reshape(ash1)
        b = b.ravel()
        if not per:
            # adjust a, b for lo and hi constraints, if any
            if nstrip:
                # strip unused constraints on final interval
                a = a[:, :-nstrip]
                b = b[:-nstrip]
            if hi:
                # fill in b values for hi constraints
                dxbc = dxscl[-1]
                for ibc in range(nhi):
                    b[ibc-nhi] = hi[ibc][i] * dxbc[ibc]
            if lo:
                # no vestigial constraint equations on lo side, add them
                a = concatenate((alo, a), axis=1)
                b = concatenate((zblo, b))
                dxbc = -dxscl[0]  # apply minus sign here
                for ibc in range(nlo):
                    b[ibc] = lo[ibc][i] * dxbc[ibc]

            # solveh_banded only works for positive definite a matrix.
            # The constraint unknowns and equations may invalidate this,
            # so even though a is symmetric, it may not be positive.
            yi = solves_banded(a, b, lower=True, check_finite=False,
                               overwrite_ab=True, overwrite_b=True)
        else:
            nstrip = 0
            yi = solves_periodic(a, b, lower=True, check_finite=False,
                                 overwrite_ab=True, overwrite_b=True)
        c.append(yi)
    c = array(c)
    # The coefficients c include the Lagrange multipliers.  Disentangle.
    mask = zeros(ash0[1:], dtype=bool)
    mask[:, :n1] = True  # mark the coefficients
    mask = mask.reshape(ash1[1:])
    if nstrip:
        mask = mask[:-nstrip]
    if nlo:
        mask = concatenate((zeros(zblo.shape, dtype=bool), mask))
    if cost:
        cc = -2.*c[:, ~mask]  # variables were -lambda/2
    c = c[:, mask]
    ny = y.shape[0]
    pwp = c.reshape(ny, nk1, n1) / dxscl  # restore dx units
    c = zeros((n1, ny, nk1+2), dtype=dtype)
    c[:, :, 1:-1] = transpose(pwp, (2, 0, 1))
    c[:, :, 0] = c[:, :, 1]
    c[:, :, -1] = polyddx(c[:, :, -2], dxk[-1])
    if not per:
        c[extrap[0]+1:, ..., 0] = 0
        c[extrap[1]+1:, ..., -1] = 0
    c = c.reshape((n1,)+yshape+(nk1+2,))
    pwp = PerPwPoly.new(xk, c) if per else PwPoly.new(xk, c)
    if cost:
        if not per:
            ihi = nlo + nc1*(nk1-1)
            costl, cost, costh = cc[:, :nlo], cc[:, nlo:ihi], cc[:, ihi:]
            costl *= dxscl[0, :nlo]
            cost = cost.reshape(ny, nk1-1, nc1) * dxscl[:-1, :nc1]
            costh *= dxscl[-1, :nhi]
            cost = transpose(cost, (2, 0, 1)).copy().reshape((nc1,)+yshape
                                                             + (nk1-1,))
            costl, costh = costl.T.copy(), costh.T.copy()
            costl = costl.reshape((nlo,)+yshape)
            costh = costh.reshape((nhi,)+yshape)
            cost = tuple(cost) + tuple(costl) + tuple(costh)
        else:
            cost = cc.reshape(ny, nk1, nc1)
            cost *= roll(dxscl[:, :nc1], -1, axis=0)
            cost = transpose(cost, (2, 0, 1)).copy().reshape((nc1,)+yshape
                                                             + (nk1,))
            cost = tuple(cost)
        return pwp, cost
    return pwp


def _bico(deg, nc=None, dtype=float):
    """Return binomial coefficients up to degree deg."""
    if nc is None:
        nc = deg - 1
    c = zeros((nc+1, deg+1), dtype=dtype)
    c[0] = 1
    ci = c[0]
    for i, cj in enumerate(c[1:]):
        cj[i+1:] = ci[i:-1].cumsum()
        ci = cj
    return c


def _splfit_setup(xk, x, y, lo=(), hi=(), per=None, extrap=None):
    # handle non-tuple endpoint values as a convenience
    if not isinstance(lo, tuple):
        lo = (lo,)
    if not isinstance(hi, tuple):
        hi = (hi,)
    if per and (lo != () or hi != () or (extrap is not None)):
        raise TypeError("periodic bcs preclude lo, hi, or extrap")
    xk, x, y = map(asfarray, (xk, x, y))
    if xk.ndim != 1 or xk.size < 2:
        raise TypeError("xk must be 1D array with at least two points")
    if xk[0] > xk[-1]:   # handle reversed xk as convenience
        xk = xk[::-1]
        lo, hi = hi, lo
        try:
            extrap = extrap[::-1]
        except (TypeError, IndexError):
            pass
    xkorig = xk.copy()
    if per:
        x0 = xk[0]
        x = (x - x0)%(xk[-1] - x0) + x0
    else:
        xmin, xmax = x.min(), x.max()
        if xmin < xk[0]:
            if lo:
                raise ValueError("lo illegal if any x beyond xk[0]")
            cats = ([xmin], xkorig)
        else:
            cats = (xkorig,)
        if xmax > xk[-1]:
            if hi:
                raise ValueError("hi illegal if any x beyond xk[-1]")
            cats += ([xmax],)
        xk = concatenate(cats) if (len(cats) > 1) else xkorig
    return xkorig, xk, x, y, lo, hi, extrap


def _splfit_args(xk, x, y, sigy=None, lo=(), hi=()):
    ndimx = x.ndim
    ndimy = y.ndim - ndimx
    y = y + zeros_like(x)
    x = x + zeros(y.shape[ndimy:], dtype=y.dtype)
    yshape = y.shape[0:ndimy]
    if sigy is None:
        sigy = ones_like(y)
    else:
        sigy = asfarray(sigy) + zeros_like(y)
    x = x.ravel()
    shape = (yshape if ndimy else (1,)) + (x.size,)
    y = y.reshape(shape)
    sigy = sigy.reshape(shape)
    # y, sigy always 2D, x always 1D
    # ix is the interval index, 0 for first interval, xk.size-2 for last
    ix = minimum(maximum(xk.searchsorted(x), 1), xk.size-1) - 1
    if lo or hi:
        dtype = y.dtype
        yzero = zeros(yshape, dtype=dtype)
        olo, ohi, lo, hi = lo, hi, [], []
        for bc in olo:
            lo.append((asfarray(bc).astype(dtype) + yzero).ravel())
        for bc in ohi:
            hi.append((asfarray(bc).astype(dtype) + yzero).ravel())
    return x, y, sigy, ix, xk[1:]-xk[:-1], yshape, lo, hi

########################################################################
# workhorse functions


def polyfun(c, x):
    """Evaluate polynomial c[0] + c[1]*x + c[2]*x**2 + ...

    Parameters
    ----------
    c : array_like
        Polynomial coefficients in order of increasing power of `x` are in
        first dimension.  Trailing dimensions, if any, must be conformable
        with `x`.
    x : array_like
        Points at which to evaluate polynomial.

    Returns
    -------
    p : ndarray
        Polynomial values, p = c[0] + c[1]*x + c[2]*x**2 + ...
    """
    c, x, p = _polysetup(c, x)
    for cn in c[-3::-1]:
        p *= x   # modify p in place to minimize memory allocations
        p += cn
    return p


def polyddx(c, x, nd=None, norm=False):
    """Evaluate polynomial c[0] + c[1]*x + c[2]*x**2 + ... and its derivatives.

    Parameters
    ----------
    c : array_like
        Polynomial coefficients in order of increasing power of `x` are in
        first dimension.  Trailing dimensions, if any, must be conformable
        with `x`.
    x : array_like
        Points at which to evaluate polynomial.
    nd : int, optional
        Number of derivatives to compute, defaults to the degree of `c`,
        ``c.shape[0]-1`` (that is, all non-zero derivatives).
    norm : bool, optional
        If false (default), returns N-th derivative divided by N! (Taylor
        series expansion coefficient).  If true, returns N-th derivative.

    Returns
    -------
    pdp : ndarray
        Polynomial and derivative values for::

            p = c[0] + c[1]*x + c[2]*x**2 + ...
            dpdx = c[1] + 2*c[2]*x + 3*c[3]*x**2 + ...
            d2pdx2 = 2*c[2] + 6*c[3]*x + 12*c[4]*x**2 + ...
            ...
            pdp = [p, dpdx, d2pdx2/2, ..., dNpdxN/N!]

        Note that polyddx returns the N-th derivative divided by N!, that
        is, the coefficient of the ``(y-x)**N`` term in the Taylor series
        expansion of the polynomial around the point `x`.  If you want the
        true N-th derivative, set the norm argument.

    Note that ``cx = polyddx(c, x)`` are the coefficients of the polynomial
    translated by x.  That is, ``polyddx(polyddx(c, x), y-x)`` is the
    same as ``polyddx(c, y)``.
    """
    c, x, p = _polysetup(c, x)
    shape = c.shape
    if nd is None:
        nd = shape[0] - 1
    # prepend axis for derivative order 0,1,2,..,nd in result
    p = concatenate((p[newaxis], zeros((nd,)+p.shape)))
    q = p[0:nd]  # crucial that this is a view into p, not a copy
    if nd:
        p[1] = c[-1]
    cns = zeros_like(p)
    for cn in c[-3::-1]:
        cns[0], cns[1:] = cn, q   # shift cn into cns
        p *= x   # modify p in place so view q remains valid
        p += cns
    if norm and nd:
        p[1:] *= arange(1, nd+1).cumprod().reshape((nd,)+(1,)*(p.ndim-1))
    return p


def _polysetup(c, x):
    c, x = asarray(c), asarray(x)
    try:
        p = c[-2] + c[-1]*x
    except IndexError:
        p = c[-1] + zeros_like(x)
    return c, x, p

########################################################################
# useful special cases

#heaviside = PwPoly([0.], [0., 1.])
#absval = (2.*heaviside - 1.).integ()
