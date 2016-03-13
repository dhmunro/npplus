# pwpoly.py
# piecewise polynomial class PwPoly

from numpy import array, asarray, asfarray, zeros, zeros_like, ones, arange
from numpy import promote_types, eye, concatenate, searchsorted, einsum, roll
from numpy import newaxis, maximum, minimum, absolute, any, isreal, real
from numpy import prod, isclose, allclose, transpose
from numpy.linalg import inv
from scipy.linalg import solve_banded, solve

class PwPoly(object):
    """Piecewise polynomial function.

    Typical usage::

        pwp = PwPoly(xk, yk, dydxk)  # define a piecewise cubic function
        y = pwp(x)                   # evaluate function at x
        y, dydx, d2ydx2 = pwp(x, 2)  # evaluate function and 2 derivatives

    The points xk are called "knot points" of the piecewise polynomial.
    Outside the endpoints of xk, the function will have the degree to
    which it is continuous at the other endpoints.

    The PwPoly constructor produces a smooth fit that is local, in the
    sense that the function in the interval between consecutive knot
    points depends only on the given function and derivative values at the
    interval endpoints.  The spline function constructs and returns a PwPoly
    which is smoother (for a given degree) by using the function values you
    provide at all the knot points to determine the function within each
    interval.  The bspline function constructs a PwPoly using a third kind
    of input data, in which the points you specify do not lie on the curve
    at all (unlike spline and the PwPoly constructor), but are merely
    "control points" that guide the curve.  Finally, the pwfit function
    constructs a PwPoly that is the statistical "best fit" to a cloud of
    data points you provide.

    Parameters
    ----------
    xk : array_like
        List of abcissa values bounding the pieces, in increasing order.
        As a convenience, strictly decreasing xk are also accepted; the
        constructor will reverse both the xk and args.
    *args : each array_like
        Each is a list of function and derivative values yk, dydxk, d2ydx2k,
        etc. corresponding to the points xk.  This produces an odd degree
        polynomial in each interval, continuous to the specified derivatives
        at the xk points.  To get even degree polynomials, specify the final
        derivative with one fewer point than xk -- this produces a polynomial
        of degree P=2*N continuous at xk up to the N-1 derivatives you
        specified at all xk points, and with the P-th (not N-th) derivative
        equal to the final values specified.  Finally, for complete generality,
        you can specify every argument with one more point than xk.  In this
        case, the arguments simply become the coefficients of the polynomials
        in each interval with no continuity guaranteed at the xk.  Note that
        the origin for coefficients in each interval is the first point of
        the interval (the smaller), except for the semi-infinite interval
        before the first knot point, which is relative to the first knot.

    All the ``*args`` may have a set of leading axes to define a
    multidimensional piecewise polynomial function of ``xk``.  That is,
    the interpolation direction is the last axis of xk, yk, dydxk, etc.

    Let p and q be PwPoly instances.  The following "natural" operators work:

    Operators
    ---------
    p(x) : evaluate p at x, returning array of same shape as x
        If p is multidimensional, its additional dimensions are leading
        dimensions of the result.
    p+q, p-q, -p, p*q : return a new PwPoly instance
        If p and q have different xk, result is on union of points.  Either
        p or q may be scalars (or arrays same shape as p(0)).
        For array_like q, division p/q also works.
    p[i] : components of p, only for multidimensional p
    len(p) : first dimension length of p, only for multidimensional p

    Attributes
    ----------
    xk : 1D ndarray of knot points
    c : coefficient array, axes are [degree, dimensionality, knots+1]
        The polynomial coefficients are for the polynomial in (x-xk[i-1]),
        where xk[i-1] <= x < xk[i], except for c[...,0], which like c[...,1]
        is for the polynomial in (x-xk[0]).

    Properties
    ----------
    ndim : number of dimensions of p(0)
    shape : shape of p(0)
    deg : degree of each polynomial piece, same as degree()

    Methods
    -------
    degree() : degree of each polynomial piece
    roots()  : x where p(x) == 0 (error unless p.ndim==0)
    deriv(m) : mth derivative
    integ(m) : mth integral
    reknot(xk, n) : estimate with new knot points
    addknots(xk) : return same function with additional knot points
    allknots(p) : find union of knot points
    jumps(n) : return discontinuous jumps at knot points

    Notes
    -----
    PwPoly is intended for low degree polynomials, usually degree 1 and 3.
    It should work tolerably well at or below degree 7, but roundoff errors
    will become significant at higher degree.  To get a better fit with
    PwPoly, use more knots, not higher degree.

    See also
    --------
    pwfit, spline, bspline
    """
    def __init__(self, xk=None, *args):
        if xk is None:
            return
        xk = asfarray(xk)
        if xk.ndim != 1:
            raise ValueError("xk must be 1D array_like")
        nk = xk.size
        if nk>1 and xk[-1]<xk[0]:
            # as a convenience, reverse xk and args so xk increasing
            xk = xk[::-1]
            args = asarray(args)[..., ::-1]
        argz = asfarray(args[-1])
        even = (argz.shape[-1] == nk-1)
        if even:
            args = args[0:-1]
            if not args:
                zero = zeros_like(argz[...,0:1])
                args = (concatenate((zero, argz, zero), axis=-1), )
                even = False
        args = asfarray(array(args))  # copy of some float type
        nda = args.ndim - 2  # additional dimensions of args
        if nda < 0:
            raise ValueError("yk and derivatives must be at least 1D")
        dtype = promote_types(xk.dtype, args.dtype)
        if args.shape[-1] != nk+1:
            if args.shape[-1] != nk:
                raise ValueError("yk trailing axis inconsistent with xk size")
            if nk < 2:
                raise ValueError("xk needs at least 2 points when yk same size")
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
            rght = einsum('ij,j...->i...', m, rght) # now h+1...2*h+1 coeffs
            dxn *= dx*dxn[-1]  # [dx**(h+1), ..., dx**(2*h+1)]
            rght *= 1./dxn
            c = concatenate((left, rght, argz))
            # add extrapolation coefficients - just the final given ones
            left = zeros_like(c[..., 0:1])
            c = concatenate((left, c, left), axis=-1)
            c[0:h+1, ..., 0:nk+1:nk] = args[..., 0:nk:nk-1]
            args = c
        # set up for __call__
        self.__rawinit(xk, args)  # will always concatenate and copy xk

    def __rawinit(self, xk, c):
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
        pwp : PwPoly or a subclass

        If xk.size is K+1, xk will not be copied.  If xk.size is K, xk[0]
        will be duplicated.  The coefficient array c is always used uncopied.
        No error checking is done; this is a low level method for creating
        PwPoly results, not a user interface.
        """
        pwp = cls()
        PwPoly.__rawinit(pwp, xk, c)
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
            cls._two_sided_cache[h] = m = linalg.inv(m)
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
            Points to evaluate function
        nd : optional int, default 0
            Number of derivatives to evaluate

        Results
        -------
        y : ndarray, or tuple of 1+nd ndarrays when nd>0
            Function values, or function and derivative values.  Each array
            has same same shape as x, unless the piecewise function was
            defined with additional dimensions, in which case those become
            the leading dimensions of the result arrays.
        """
        x = array(x)    # make copy here for -= below
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
        """Number of dimensions of result when evaluated at a scalar x."""
        return self.c.shape[1:-1]  # first dimension is degree, last is knot

    def deriv(self, m=1):
        """Derivative as a new piecewise polynomial.
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
        """
        c = self.c
        consts = zeros((m,) + c.shape[1:-1])
        lbnd = asfarray(lbnd)
        if lbnd.ndim:
            raise ValueError("lower bound must be scalar")
        dxlb = self.xk0 - lbnd
        if k is not None:
            k = asfarray(k)
            if k.ndim == c.ndim-2:
                k = k[newaxis]
            if k.shape[0] < m:
                k = concatenate((k, zeros((m-k.shape[0],)+k.shape[1:])))
            consts += k
        # integration constants same for every interval xk
        consts = consts[...,newaxis] + zeros(c.shape[-1])
        for k in consts[:,newaxis]: # add leading (1,) axis for concatenate
            n = c.shape[0]
            n = 1. / arange(1, n+1).reshape((n,) + (1,)*(c.ndim-1))
            c = concatenate((k, c*n))
            # c[0] each have lower bound at xk, adjust to common lbnd
            c[0] -= polyfun(c, dxlb)
        return self.new(self.xk0, c)

    def roots(self, value=0., tol=1.e-9):
        """Return the values of x for which the piecewise polynomial is zero.

        Parameters
        ----------
        value : float, optional
            return values of x for which piecewise polynomial equals value
        """
        c = self.c.copy()
        if c.ndim != 2:
            raise ValueError("cannot find roots of multidimensional function")
        if value:
            c[0] -= value
        xk, xk0 = self.xk, self.xk0
        dx = xk - xk0[:-1]
        yhi = polyfun(c[:,:-1], dx)
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
        dxn = dx + zeros((n+1,)+dx.shape)
        dxn[0] = 1
        dxn = dxn.cumprod(axis=0)  # [1, dx, dx**2, ..., dx**n]
        c *= dxn  # scale each interval to (0,1)
        deg = absolute(c)
        deg = (deg > tol*deg.max(axis=0)) * arange(n+1)[:,newaxis]
        deg = deg.max(axis=0)  # effective degree in each interval
        x = [x]
        nold, imax = -1, len(xk)
        for i, ci in enumerate(c.T):  # loop on knot intervals
            n = deg[i]
            if n < 1: continue
            if n != nold:
                m = zeros((n, n), dtype=c.dtype)
                m.reshape(n*n)[n::n+1] = 1
            m[:, -1] = -ci[:-1] / ci[-1]
            xi = linalg.eigvals(m)
            xok, xi = isreal(xi), real(xi)
            if i == 0:
                xok &= (xi <= 0.)
            elif i < imax:
                xok &= (xi >= 0.) & (xi <= 1.)
            else:
                xok &= (xi >= 0.)
            if not any(xok): continue
            x.append(xi[xok]*dx[i] + xk0[i])  # undo (0,1) dx c scaling
        x = concatenate(x)
        x.sort()
        # remove duplicates or near duplicates
        if len(x) > 1:
            mask = ones(x.shape, dtype=bool)
            i = xk.searchsorted(x)
            dx = dx[i]
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
            maximum degree of returned coefficients, all by default

        Results
        -------
        dc : ndarray (N,...,K)
            Degree is first axis, dc[0] is jump in function value,
            dc[1] is jump in derivative, dc[2] is half jump in second
            derivative, and so on.  Knot is final axis.  Any additional
            axes are the dimensionality of the PwPoly.
        """
        if n is None:
            n = self.c.shape[0] - 1
        dx = self.xk0[1:] - self.xk0[:-1]
        cminus = polyddx(self.c[...,:-1], dx, n)  # value just below xk
        return self.c[0:1+n,...,1:] - cminus

    def reknot(self, xk, n=None):
        """Return a new PwPoly with knot points xk that approximates this one.

        If n is the polynomial degree, the resulting piecewise polynomial
        will match (n-1)/2 derivatives of the old one at the points xk.
        If n is even, the n-th derivative at the midpoint of each xk interval
        will also match the original.

        Parameters
        ----------
        xk : 1D array_like
            new knot points
        n : optional int
            degree for new piecewise polynomial, default same as this one

        Results
        -------
        p : PwPoly
        """
        if n is None:
            n = self.c.shape[0] - 1
        xk = asarray(xk)
        if xk.ndim != 1:
            raise ValueError("new knot points must be 1D array_like")
        if xk[0] > xk[-1]:
            xk = xk[::-1]  # permit decreasing xk as convenience
        h = (n - 1) // 2
        if h >= 0:
            c = self(xk, h)
            if h == 0:
                c = c[newaxis]
        else:
            c = (self.c[...,0:1]+xk)[0:0,...]
        if not (n & 1):
            # breaks if only one knot point...
            dx = 0.5*(xk[1:] - xk[:-1])
            x = xk + concatenate(dx, dx[-1:])
            x = concatenate((xk[0]-dx[0:1], x))
            ix = searchsorted(self.xk, x)
            c = concatenate((c, self.c[...,ix][-1:,...]))
        return self.new(xk, c)

    def addknots(self, xk, _nocheck=False):
        """Return a new PwPoly with knot points xk that matches this one.

        The xk must be a superset of the existing knot points for this to
        work as intended.

        Parameters
        ----------
        xk : 1D array_like
            new knot points
        n : optional int
            degree for new piecewise polynomial, default same as this one

        Results
        -------
        p : PwPoly
        """
        if not _nocheck:
            xk = self.allknots(xk)
            if xk is self.xk:
                return self
            if xk.ndim != 1:
                raise ValueError("new knot points must be 1D array_like")
            elif len(xk) == len(self.xk):
                return self
        ic = self.xk.searchsorted(0.5 * (xk[1:] + xk[:-1]))
        ic = concatenate(([0], ic, [len(self.xk)]))
        xk0 = concatenate((xk[0:1], xk))
        # ic is index of interval in self.xk for each interval bounded by xk
        c = polyddx(self.c[...,ic], xk0-self.xk0[ic])
        return self.new(xk, c)

    def allknots(self, *args, **kwargs):
        """Return union of knot points among piecewise polynomials.

        Parameters
        ----------
        *args : sequence of PwPoly

        Keywords
        --------
        tol : float, default 1e-6
            relative tolerance compared to interval between knots

        Results
        -------
        xk : 1D ndarray
            union of knot points in increasing order
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
        return {xk0: self.xk0.copy(), c: self.c.copy()}
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
            raise TypeError("scalar piecewise polynomial has no len()")
        return s[0]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        colon = slice(None)
        return self.new(self.xk0, self.c[(colon,)+key+(Ellipsis,colon)])

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
            other = asarray(other, dtype=self.c.dtype)[...,newaxis]
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
            raise ValueError("cannot divide by a PwPoly")
        return self.new(this.xk0, this.c / other)

class PeriodicPwPoly(PwPoly):
    """Periodic piecewise polynomial function.

    Like PwPoly, except first and last knot points xk are the same point,
    but advanced by one period.  Input arguments to the function will be
    reduced to this period before evaluation, so there is no extrapolation.

    Note that when you initialize a PeriodicPwPoly, you must explicitly
    supply the duplicate final point at the end of the period.
    """
    def __init__(self, *args, **kwargs):
        super(PeriodicPwPoly, self).__init__(*args, **kwargs)
        if len(self.xk) < 2:
            raise ValueError("cannot create periodic function with <2 knots")
        # set extrapolation to guard against roundoff error
        self.c[...,0] = self.c[...,1]
        self.c[...,-1] = polyddx(self.c[...,-2], self.xk[-1]-self.xk[-2])
        self.period = self.xk[-1] - self.xk[0]
    def __call__(self, x, nd=0):
        x0 = self.xk[0]
        x = x0 + (x - x0)%self.period
        super(PeriodicPwPoly, self).__call__(x, nd)

########################################################################
# B-spline curves deserve their own class orthogonal to PwPoly
# In practice, best to derive a PwPoly from the Bspline

class Bspline(object):
    """A B-spline is a clever representation of a piecewise polynomial.

    The PwPoly class is more efficient for computing function values, but
    the Bspline class may more clearly represent the function properties.
    The Bspline order is one plus the polynomial degree (that is, the
    number of polynomial coefficients).  A Bspline has the high-order
    continuity of a spline -- missing from the PwPoly constructor -- and
    the locality of the PwPoly constructor missing from a spline.  (Locality
    means that only order of the Bspline control points affect the function
    value at any given point.  In a spline, changing any of the specified
    values changes the function at nearly every point.)  What a Bspline
    gives up is the specification of the function values themselves; you
    specify only control points rather than function values.  For fitting
    problems there is no reason to prefer function values over control points.

    Parameters
    ----------
    t : 1D array_like, or int
        The knot vector, a monotonic list of numbers.  If t is a decreasing
        sequence, both t and p will be reversed.
        The t vector may contain repeated values; it need not be strictly
        monotonic, merely non-decreasing (or non-increasing).  A knot value
        that repeats m times has multiplicity m.  The piecewise polynomial
        has at least k-1-m continuous derivatives.
        Alternatively, t is the order k (degree plus one) of the B-spline.
    p : array_like
        The final index of p must have a length k less than the length of t,
        where k is the order of the B-spline.  The degree of the piecewise
        polynomial in t is k-1.  Any leading dimensions of p represent a
        B-spline curve in multiple dimensions.
    periodic : optional int
        If non-zero, p.shape[-1] must equal len(t), and periodic is the
        order (degree plus one) of the B-spline.  The first and last p
        values should be equal, so t and p represent one period with its
        first point duplicated, so that t[-1]-t[0] represents the period
        in t.

    The B-spline curve does not necessarily pass through any of the p; they
    are "control points".  The important feature of these control points is
    locality: The polynomial on the interval t[i] <= t < t[i+1] is completely
    determined by the k p-values p[...,i-k+1] through p[...,i].  The
    B-spline curve is defined only on the interval t[k-1] <= t <= t[1-k]
    (that is, omitting the first and last k-1 t intervals), because outside
    this interval, the curve depends on fewer than k p-values.  In fact,
    each p value is associated with the ranges t[j] <= t <= t[j+k].  There
    are k such ranges, namely j=i-k+1 through j=i if t is in interval i.

    A Bezier curve of order k (degree k-1) is a B-spline with 2*k knot points
    and only two values, each of multiplicity k.

    The deBoor recursion algorithm defines the piecewise polynomial in terms
    of the values of p and t:

        P[i,0](u) = p[...,i] if t[i] <= u < t[i+1] else 0
        P[i,j](u) = ((t[i+k-j]-u)*P[i-1,j-1](u) + (u-t[i])*P[i,j-1])
                    / (t[i+k-j] - t[i])

    and the function value is P[i,k-1](u) when t[i] <= u < t[i+1].

    The derivative is a B-spline of order k-1 (degree k-2) with

        pdot[...,i] = (k-1) * (p[...,i] - p[...,i-1]) / (t[i+k-1] - t[i])

    where p[...,i-1] = 0 for i=0 and p[...,i] = 0 for i=len(t)-k.  Note
    that p.shape[-1]=len(t)-k, but that pdot.shape[-1]=len(t)-k+1, because
    pdot has order k-1.
    """

########################################################################
# alternative PwPoly constructors are implemented as functions

def pwfit(x, y, xk, n=1, sigy=1.0, stats=False, given=None):
    """Construct a PwPoly that is the least squares best fit to given data.

    Parameters
    ----------
    x : array_like
    y : array_like
        The given (x,y) data points to be fit.
        The y data may have additional leading dimensions compared to x, in
        order to fit a curve in a multi-dimensional space.
    xk : 1D array_like
        The knot points for the PwPoly fit.
    n : int, optional
        The degree of the PwPoly.  Defaults to 1, a piecewise linear fit.
    sigy : array_like, optional
        Standard deviations of the points y, must be conformable with y.
        Points will be weighted as 1/sigy**2.  Without sigy (by default)
        all y values are weighted equally.
    stats : bool, optional
        If True, returns covariances and other fitting statistics in addition
        to the PwPoly fit itself.
    given : array_like
        Constraints on the fit.

    Returns
    -------
    p : PwPoly
        The PwPoly with knots xk and degree n that best fits the given(x,y).
        With stats set, also returns:

    See Also
    --------
    pline : general piecewise linear interpolator
    spline : construct spline
    bspline : construct B-spline
    pwfit : piecewise polynomial fit to scattered data
    PwPoly : piecewise polynomial class
    """

def pline(x, y, extrap=None, periodic=False):
    """Return a piecewise linear fit through given points (x,y).

    Convenience shorthand for spline(x, y, n=1).

    Note that pline(x, y) by default is constant before the first x and
    after the final x.  Use pline(x, y, 1) to extrapolate using the
    linear function in the first and last intervals.
    """
    return spline(x, y, n=1, extrap=extrap, periodic=periodic)

def spline(x, y, n=3, lo=(), hi=(), periodic=False, extrap=None):
    """Construct a PwPoly as a spline fit through given points (x,y).

    A spline is a piecewise polynomial of degree n (default 3) passing
    through all of the given points with continuous derivatives up to
    degree n-1 at every point.  Before x[0] and after x[-1], the result
    will be of degree n-1, maintaining continuity of n-1 derivatives at
    every knot point, including the first and last.

    Note that K knots define K-1 intervals, requiring a total of
    (n+1)*(K-1)=n*K-n+K-1 polynomial coefficients.  There are n*(K-2)
    continuity constraints at the interior knots and K y values,
    for a total of n*K-2*n+K constraint equations.  We require n-1
    additional constraints to determine a unique spline fit.  When
    n is odd (linear, cubic, pentic, etc.), these additional constraints
    may be divided equally between the first and last knots points.
    When n is even, there is no way to treat the first and last knots
    symmetrically.

    By default, spline forces the (n-1)//2 highest derivatives to zero
    at each endpoint, with the first endpoint getting an extra zero
    derivative in the case of even n.  The lo and hi keywords allow you
    to split up the n-1 constraints among the possible derivatives and
    endpoints however you wish.

    Parameters
    ----------
    x : 1D array_like
    y : array_like
        The given (x,y) data points to be fit.
        The y data may have additional leading dimensions, in order to fit a
        curve in a multi-dimensional space.
        The x values must be strictly increasing, although as a convenience
        spline will reverse both x and y (and swap lo and hi) if x is
        decreasing.
    n : int, optional
        The degree of the PwPoly.  Defaults to 3, a piecewise cubic spline.
        (Note that the PwPoly constructor gives the piecewise linear spline.)
    lo : tuple of array_like, optional
    hi : tuple of array_like, optional
        Boundary conditions at the endpoints x[0] and x[-1].  Each tuple
        represents the values of (dydx, d2ydx2, d3ydx3, ...) at x[0] for lo
        or x[-1] for hi.  Use None for a derivative to be not specified.
        The highest derivative you can specify is n-1, that is d2ydx2 for
        a cubic spline, dydx for a quadratic spline, and so on.  For the
        special case that you only wish to specify dydx at an endpoint,
        lo or hi need not be a tuple (dydx,); spline accepts simply dydx
        to mean (dydx, None, None, ...).  The maximum number of derivatives
        you can specify in both lo and hi is n-1, that is, 2 for a cubic
        spline, 4 for a pentic, and so on.  If you specify fewer than n-1
        derivatives, spline will set the highest unspecified derivatives
        to zero, begining with lo and alternating lo, hi, lo, etc., until
        n-1 derivative are specified.  For example, if you specify neither
        lo nor hi for a cubic spline, then spline sets d2ydx2=0 at both
        endpoints, and the "natural spline" with zero curvature at the
        endpoints is the default.  This is equivalent to lo=(None,0.),
        hi=(None,0.).  Note that the derivatives are broadcast to the leading
        dimensions of y.
    periodic : bool
        True to make all derivatives match at first and last points of x.
        This produces a PeriodicPwPoly function, which maps any inputs
        outside the first and last points of x into that interval.
    extrap : int or (int,int)
        Degree to maintain continuity beyond endpoints, can be set separately
        for before first and after last interval.

    See Also
    --------
    pline : construct polyline
    bspline : construct B-spline
    pwfit : piecewise polynomial fit to scattered data
    PwPoly : piecewise polynomial class
    """
    # any way to generate other members of family with different BCs?
    # yes -- just pass 0*y and desired BCs
    x, y = asfarray(x), asfarray(y)
    dtype = promote_types(x.dtype, y.dtype)
    x, y = x.astype(dtype), y.astype(dtype)
    nk = len(x)
    if x.ndim!=1 or nk<2:
        raise ValueError("x must be 1D array_like with at least two points")
    shape = y.shape
    if shape[-1] != nk:
        raise ValueError("y must have same final axis as x")
    shape = shape[:-1]
    nm1 = n - 1
    if extrap is None:
        extrap = (nm1, nm1)
    elif not isinstance(extrap, tuple):
        extra = (extrap, extrap)
    if x[0] > x[-1]:  # as a convenience, permit monotonic decreasing x
        x, y = x[::-1].copy(), y[...,::-1].copy()
        lo, hi = hi, lo
        extrap = extrap[::-1]
    if not isinstance(lo, tuple):  lo = (lo,)
    if not isinstance(hi, tuple):  hi = (hi,)
    if len(lo)>nm1 or len(hi)>nm1:
        raise ValueError("cannot specify more than n-1st derivative in bc")
    if n < 2:
        p = PeriodicPwPoly(x, y) if periodic else PwPoly(x, y)
        xk, c = p.xk, p.c
        c[...,0] = c[...,1]
        c[extrap[0]+1:,...,0] = 0
        if len(xk) > 1:
            c[...,-1] = polyddx(p.c[...,-2], xk[-1]-xk[-2])
            c[extrap[1]+1:,...,-1] = 0
        return p
    if periodic and (lo or hi or extrap):
        raise ValueError("periodic boundary conditions preclude lo or hi")
    else:
        nlo = len(lo) - lo.count(None)
        nhi = len(hi) - hi.count(None)
        missing = nm1 - (nlo+nhi)
        if missing:
            # insert default boundary conditions where unspecified
            lo = list(lo + (None,)*(nm1 - len(lo)))
            hi = list(hi + (None,)*(nm1 - len(hi)))
            i = n - 2
            while missing:
                if i < 0:
                    raise IndexError("impossible error, logic bug?")
                if lo[i] is None:
                    lo[i] = 0.    # fill in missing BC with zero
                    missing -= 1
                if missing and hi[i] is None:
                    hi[i] = 0.    # fill in missing BC with zero
                    missing -= 1
                i -= 1
            lo = tuple(lo)
            hi = tuple(hi)
        if periodic:
            # periodicity "truth" test logic unclear, so just make it so
            y = y.copy()
            y[...,-1] = y[...,0]
    nlo = len(lo)
    nhi = len(hi)
    one = ones((), dtype=dtype)
    m = polyddx(eye(1+n, dtype=dtype), one)[:-1,1:]
    # [[  1.,   1.,   1.,   1.,   1.],
    #  [  1.,   2.,   3.,   4.,   5.],
    #  [  0.,   1.,   3.,   6.,  10.],
    #  [  0.,   0.,   1.,   4.,  10.],
    #  [  0.,   0.,   0.,   1.,   5.]]  for example, when n=5
    m = concatenate((m, zeros_like(m)), axis=1).ravel()
    nun = (nk-1)*n        # number of unknowns
    diags = one.repeat(nun)  # begin with subdiagonal
    diags[nm1::n] = 0     # 1 1 1 1 0 1 1 1 1 0 ... 1 1 1 1    when n=5
    diags = [diags]
    strd = n+n+1
    for i in range(nm1):  # continue with diagonal and superdiagonals
        _ = m[i:i+n*strd:strd][newaxis].repeat(nk,axis=0).ravel()[1:nun]
        diags.append(roll(_, i, axis=-1))
    dx = x[1:] - x[:-1]     # nk-1 differences
    if nk > 2:
        # matching at knot between intervals needs powers of interval ratio
        dxr = dx[1:] / dx[:-1]  # nk-2 ratios of intervals
        dxr = dxr[:,newaxis].repeat(n, axis=1)
        dxr[:,0] = -1
        dxr = dxr.cumprod(axis=1)
        dxr[:,0] = 1   # (nk-2)*[1,-dxr,-dxr**2,...,-dxr**(n-1)]
    else:
        dxr = array([], dtype=dtype)
    diags.append(concatenate((dxr.ravel(), dlo[0:1])))  # last superdiagonal
    zero = zeros(shape)
    rhs = zero[...,newaxis].repeat(nun, axis=-1)
    rhs[...,0::n] = y[...,1:] - y[...,:-1]
    # translate BCs into matrix (row, RHS) pairs, exactly n-1 pairs total
    lo, rhsl = _spline_get_bc(lo, zero, x[1]-x[0], nun)
    hi, rhsh = _spline_get_bc(hi, zero, x[-1]-x[-2], nun, (nk-2)*n)
    diags = array(lo + diags + hi)
    zero = zero[...,newaxis]
    rhs = concatenate((zero.repeat(nlo,axis=-1), rhs, zero.repeat(nhi,axis=-1)))
    rhs[...,0:nlo] = rhsl
    if nhi: rhs[...,-nhi:] = rhsh
    # note: numpy.linalg.solve does multiple solves since numpy 1.4
    #  scipy.linalg.solve_banded since before 0.7
    if shape:
        # solve_banded wants additional y dimensions last
        rhs = rhs.reshape(prod(shape), nun).T.copy()
    nlnu = (1+nlo, nm1-nlo)
    rhs = solve_banded(nlnu, m, rhs, overwrite_ab=(not periodic),
                       overwrite_b=True, check_finite=True)
    if periodic:
        # So far, we have natural spline solution with highest derivatives
        # at first and last knots equal zero.  Now solve each BC=1 with all
        # others =0 to derive n-1 more conditions for continuity of the
        # function.  (All dy are zero in these solves.)
        b = zeros(nun, nm1, dtype=dtype)
        for i in range(nm1):
            if i < nlo:
                b[i, i] = 1
            else:
                b[i-nm1, i] = 1
        b = solve_banded(nlnu, m, b, overwrite_ab=True,
                         overwrite_b=True, check_finite=True)
        # translate b in last interval to final knot point
        bhi = concatenate((zeros_like(b[0:1,:]), b[-nm1:,:]), axis=0)
        bhi = polyddx(bhi, one)[1:]
        dxr = (dx[0]/dx[-1]).repeat(nm1).cumprod()
        m = dxr.dot(bhi) - b[0:nm1,:]
        # compute discontinuities in natural spline solution
        db = concatenate((zeros_like(rhs[0:1,...]), rhs[-nm1:,...]), axis=0)
        db = dxr.dot(polyddx(db, one)[1:]) - rhs[0:nm1,...]
        db = solve(m, -db, overwrite_a=True, overwrite_b=True,
                   check_finite=False)
        # m.dot(db) + drhs = 0 is periodic BC, adjust rhs accordingly
        rhs += b.dot(db)
    rdxn = 1./dx[:,newaxis].repeat(n, axis=1).cumprod(axis=1).ravel()
    rhs = rdxn.dot(rhs)
    rhs = transpose(rhs.reshape(nk-1, n, prod(shape)), (1,2,0))
    rhs = rhs.reshape((n,) + shape + (nk-1,))
    c = concatenate((rhs[0:1,...], rhs), axis=0)
    c = concatenate((c, c[...,-1:]), axis=-1)
    c[0,...] = y
    c = concatenate((c[...,0:1], c), axis=-1)
    c[...,-1] = polyddx(c[...,-2], dx[-1])
    if not periodic:
        c[extrap[0]+1:,...,0] = 0
        c[extrap[1]+1:,...,-1] = 0
    return PeriodicPwPoly(x, c) if periodic else PwPoly(x, c)

def _spline_get_bc(bc, zero, dx, nun, shift=0):
    rb = ([], [])
    if bc is not None:
        dtype = zero.dtype
        row = roll(array([1., 0.], dtype=dtype).repeat([1,nun-1]), shift)
        dxn = array(dx, dtype=dtype)
        for ybnd in bc:
            if ybnd is not None:
                rb[0].append(row)
                rb[1].append(asarray(ybnd,dtype=dtype)*dxn+zero)
            row = roll(row, 1)
            dxn *= dx
    return rb

def bspline(x, y, n=3, s=None, lo=None, hi=None, periodic=None):
    pass

# Also expose these as static constuctor methods of the PwPoly class,
# so that simply importing PwPoly (or having an instance) gives access.
# Arguably, such constructors should be class methods, but this seems
# straightforward enough.
PwPoly.fit = staticmethod(pwfit)
PwPoly.spline = staticmethod(spline)
PwPoly.bspline = staticmethod(bspline)

########################################################################
# workhorse functions

def polyfun(c, x):
    """Evaluate polynomial c[0] + c[1]*x + c[2]*x**2 + ...

    Parameters
    ----------
    c : array_like
        Polynomial coefficients in order of increasing power of x are in
        first dimension.  Trailing dimensions, if any, must be conformable
        with x.
    x : array_like
        Points at which to evaluate polynomial(s).

    Results
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
        Polynomial coefficients in order of increasing power of x are in
        first dimension.  Trailing dimensions, if any, must be conformable
        with x.
    x : array_like
        Points at which to evaluate polynomial(s).
    nd : optional int
        Number of derivatives to compute, defaults to the degree of c,
        ``c.shape[0]-1`` (that is, all non-zero derivatives).
    norm : optional bool
        If false (default), returns N-th derivative divided by N! (Taylor
        series expansion coefficient).  If true, returns N-th derivative.

    Results
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
        expansion of the polynomial around the point x.  If you want the
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
        p[1:] *= arange(1,nd+1).cumprod().reshape((nd,)+(1,)*(p.ndim-1))
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

heaviside = PwPoly([0.], [0.,1.])
absval = (2.*heaviside - 1.).integ()
