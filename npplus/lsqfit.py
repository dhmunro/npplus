# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Linear and non-linear least squares fitters.

* For a simple linear least squares fit::

      p = regress(data, mod0, mod1, mod2, ..., errs=errs)

  returns `p` model coefficients for best linear fit to data of form ::

      data ~ p[0]*mod0 + [1]*mod1 + p[2]*mod2 + ...``

  Use the ``model=1`` keyword to return more information, such as covariances.

* For a non-linear least squares fit to a parametrized function::

      model = levmar(data, f, p0, arg0, arg1, ..., errs=errs)``

  returns callable model with best fit to data of form ::

      data ~ model(arg0, arg1, ...) = f(p, arg0, arg1, ...)

  ``model.p`` are the best fit parameters, ``model.pcov`` their covariances.

--------
"""

__all__ = ['regress', 'levmar', 'LevmarError']

from numpy import array, asfarray, zeros_like, absolute, maximum, zeros, ones
from numpy import inner, ones_like, fill_diagonal, minimum, diag, sqrt, asarray
from numpy.linalg import solve, inv
from scipy.linalg import svd
from scipy.special import gammaincc
from inspect import isgeneratorfunction


def regress(data, *mdl, **kwargs):
    """Least squares fit a linear model to data. ::

        p = regress(data, m1, m2, m3, ...)

    finds 1D array of coefficients p such that::

        einsum('i,i...', p, [m1, m2, m3, ...])   approximates   data
        or simply p.dot([m1, m2, m3, ...]) if data is 1D

    as nearly as possible in a least squares sense.  You can use regress
    in underdetermined systems, that is, with more coefficients than
    data points, in which case it returns the exact solution p with
    the smallest Euclidean norm in p-space.  The ``model=1`` statistics will
    not be useful in this case.

    Parameters
    ----------
    data : array_like
        The data to be fit.
    m1,m2,m3,... : array_like
        Each of the `mi` must be conformable with data, representing
        a particular component of the linear model to explain the data.
    errs : array_like, optional keyword
        If provided, must be conformable with data, representing the
        standard deviation of each data point, default 1.
    model : bool, optional keyword
        If not provided or false, the return value is the coefficients `p`.
        If true, the return value is a ModelFit instance.
    rcond : float, optional keyword
        Reciprocal condition number, default 1e-9.  Often the data do
        no permit a definitive choice of model -- entire p-subspaces
        may produce indistinguishably good fits to the data, which
        makes the least squares matrix singular.  The rcond parameter
        determines how small a singular value you are willing to
        consider relative to the largest singular value.  Directions
        with smaller singular values are ignored when inverting the
        matrix.

    Returns
    -------
    p : ndarray or ModelFit
        If the model parameter is not provided or false, `p` are
        coefficients of the best fit model.  The length and order of `p`
        corresponds to the number and order of `mi` arguments.

        If the model keyword is true, regress returns a ModelFit
        object with many attributes and methods, including::

            model.p --> the model coefficients (the model=0 return)
            model.pcov --> covariances of p
            model.chi2 --> chi squared per degree of freedom
            model.chi2pcov --> pcov if no errs supplied in foot
            model.ndof --> number of degrees of freedom in fit
            model.s --> singular values for the model matrix
            model.u --> u[i] is unit vector in p-space corresponding to s[i]
            model() --> data produced by best fit model

    See Also
    --------
    levmar : non-linear least squares fitter
    ModelFit : class for best fit models

    Notes
    -----
    The best fit polynomial of degree three has coefficients:

        ``p = regress(y, 1., x, x**2, x**3)``

    To find the best fit to a set of data points ``(x,y)`` by any function
    of the form ``(a + b*x + c*exp(-x**2)*cos(x))``:

        ``p = regress(y, 1., x, exp(-x**2)*cos(x))``
    """
    data = asfarray(data)
    nc = len(mdl)
    m = zeros((nc,)+data.shape, dtype=data.dtype)
    for i, mi in enumerate(mdl):
        m[i] += mi
    data = data.ravel()
    m = m.reshape(nc, data.size)
    rstdev = 1. / kwargs.pop('errs', 1.0)
    rcond = kwargs.pop('rcond', 1.e-9)
    rcond = minimum(maximum(rcond, 1.e-13), 0.5)

    b = data * rstdev
    m *= rstdev
    # solve p.dot(m) = b in least squares sense using SVD
    # SVD: m = (u*s).dot(v[:s.size]), u and v orthogonal
    u, s, v = svd(m, full_matrices=False)
    v = v.dot(b)
    mask = s > rcond*s[0]  # set where kept
    rs = zeros_like(s)
    rs[mask] = 1. / s[mask]
    pc = u * rs
    p = pc.dot(v)
    if not kwargs.pop('model', False):
        return p

    # with model=1 return full statistics of the fit as a ModelFit
    s[~mask] = 0
    ndof = b.size - mask.sum()
    fit = p.dot(m)
    chi2 = fit - b
    chi2 = (chi2*chi2).sum() / maximum(ndof, 1.)
    fit /= rstdev

    def f(p):
        return fit

    pc = inner(pc, pc)
    return ModelFit(f, p, pc, chi2, ndof, u=u, s=s,
                    info=dict(src='regress'))


def levmar(data, f, p0, *args, **kwargs):
    """Least squares fit a non-linear model to data.

        ``model = levmar(data, f, p0, x)``

    is the function ``f(p, x)`` for the particular parameters `p` such that

        ``model(x)``   approximates   `data`

    as nearly as possible in a least squares sense.  The model object
    has attributes you can use to retrieve the best fit parameter
    values, convariances, and other information about the fit.

    If you can calculate the partial derivatives of `f` with respect
    to `p`, you should write `f` as a generator function with two
    yield statements.  The first yield statement must return the
    function value, while the second and final yield must return the
    partial derivatives `dfdp`.  The shape of `dfdp` must match those
    of `f`, which is the shape of data, plus a leading axis with the
    size and order of the parameter vector `p`.

    If `f` is not a generator function, levmar will estimate its partial
    derivatives using finite differences.  You may need to provide `prel`
    and `pabs` keywords to get appropriate step sizes for the finite
    differences of your parameters.

    Parameters
    ----------
    data : array_like
        The data to be fit.

    f : function or generator function
        A parametric function ``f(p, a1, a2, ..., k1=k1, k2=k2, ...)``
        representing the model for the data.  The first argument to `f`
        must be a 1D array of parameters.  Any remaining positional
        or keyword arguments to `f` represent the independent variables
        of the model.  The result of `f` must have the same dimensions
        as data.

        `f` may also be a generator function with two yield statements
        returning `dfdp` on the second as described above.

    p0 : array_like
        The initial guess for the 1D parameters `p` needed to fit data.

    a1,a2,... : arbitrary
        The model function `f` may have any number of postional
        arguments beyond its first argument `p`; any additional
        positional arguments to levmar will be passed unexamined to
        every call of `f`.

    errs : array_like
        If provided, must be conformable with `data`, representing the
        standard deviation of each data point, default 1.

    pfix : int or sequence of int
        An index or indices into `p` which are to remain fixed at their
        values in `p0`.

    pmin, pmax : array_like
        Minimum and maximum bounds for `p`.  If provided, must be
        conformable with `p`.  The function `f` will not be called with
        `p` values outside these bounds.

    prel, pabs : array_like
        Relative and absolute step sizes for computing finite difference
        partial derivatives of `f`.  These must be conformable with `p`.
        The step size is ``maximum(prel*absolute(p), pabs)``.  The
        default values are ``prel=1.e-6`` and ``pabs=1.e-9``.

        If `f` is a generator, these are ignored.

    cfg : dict
        Configuration options for levmar, overriding the default options
        in ``levmar.config``.

    quiet : bool
        If levmar exceeds ``levmar.config.itmax`` iterations, indicating
        a failure to converge, it prints a warning message unless this
        keyword is present and True.

    k1=k1,k2=k2,... : arbitrary
        Any other keyword arguments are passed unexamined to every call
        of `f`.

    Returns
    -------
    model : ModelFit
        The best fit model for the data.  This callable object will
        invoke `f` with the best fit parameters, as well as permitting you
        to inspect the parameter values themselves, their covariances, and
        other information about the fit.

    See Also
    --------
    regress : linear least squares fitter
    ModelFit : class for best fit models

    Notes
    -----
    You should scale the `p` values so that their values will not be
    very large or small.  This prevents problems with ill-conditioned
    partial derivative matrices.

    ``levmar.config`` function attribute
        Options to control the Levenberg-Marquardt algorithm::

            itmax = maximum number of gradient recalculations (100)
            tol = tolerance, stop when chi2 changes by less than to (1.e-7)
            lambda0 = initial value of L-M lambda parameter (0.001)
            lambdax = maximum permitted value for lambda (1.e12)
            gain = factor by which to change lambda (10.)
            prel = relative step size for numerical_partials (1.e-6)
            pabs = absolute step size for numerical_partials (1.e-9)

        You may set ``levmar.config`` to change the default settings, or
        use the `cfg` keyword to change any subset for a single call.
    """
    data, p0 = asfarray(data), asfarray(p0)
    errs = kwargs.pop('errs', 1.0) + zeros_like(data)
    data, errs = data.ravel(), errs.ravel()
    pfix = [kwargs.pop(p, None) for p in ('pfix', 'pmin', 'pmax',
                                          'prel', 'pabs')]
    pfix, pmin, pmax, prel, pabs = pfix
    cfg = levmar.config.copy()
    if 'cfg' in kwargs:
        cfg.update(kwargs.pop('cfg'))
    quiet = kwargs.pop('quiet', False)
    isgenf = isgeneratorfunction(f)
    if not isgenf:
        if prel is None:
            prel = cfg['prel']
        if pabs is None:
            pabs = cfg['pabs']
    z = zeros_like(p0)
    pmin, pmax, prel, pabs = [(p if p is None else z+p) for p in
                              (pmin, pmax, prel, pabs)]
    p0, pmin, pmax, prel, pabs = [(p if p is None else p.ravel()) for p in
                                  (p0, pmin, pmax, prel, pabs)]

    pfull = p0.copy()
    pmask = ones(p0.shape, dtype=bool)
    if pfix is not None:
        pmask[asarray(pfix).ravel()] = False
    p = p0[pmask]
    if pmin is not None:
        pmin = pmin[pmask]
    if pmax is not None:
        pmax = pmax[pmask]
    ndof = data.size - pmask.sum()  # number of degrees of freedom in fit
    if ndof <= 0:
        ValueError("fewer data points than model parameters")

    if isgenf:
        def g(p):
            pfull[pmask] = p
            f0 = f(pfull, *args, **kwargs)
            yield next(f0)
            yield next(f0)[pmask]  # no easy way to pass pmask to f
    else:
        def g(p):
            pfull[pmask] = p
            f0 = f(pfull, *args, **kwargs)
            yield f0
            yield numerical_partials(f, pfull, f0, pmin, pmax,
                                     prel, pabs, pmask, args, kwargs)

    itmax, tol = cfg['itmax'], cfg['tol']
    lambda0, lambdax, gain = cfg['lambda0'], cfg['lambdax'], cfg['gain']

    wgt = 1. / (errs * errs)  # chi2 weights
    lamda = lambda0
    neval = niter = 0
    while True:
        # get function value and partial derivatives
        dm, dmdp = g(p)
        neval += 1
        dm = data - dm  # residual differences between data and model
        beta = wgt * dm
        if not niter:  # must initialize chi2
            chi2 = chi20 = (beta * dm).sum()
        beta = dmdp.dot(beta)
        alpha = inner(wgt*dmdp, dmdp)
        if not niter:
            amult = ones_like(alpha)
        if not chi2:
            break

        # lambda >> 1 is steepest descents, step proportional to 1/lambda
        # lambda << 1 is linearized least squares solution
        chi2p, p1 = chi2, p
        while True:
            fill_diagonal(amult, 1.+lamda)
            p = p1 + solve(alpha*amult, beta)
            if pmin is not None:
                p = maximum(p, pmin)
            if pmax is not None:
                p = minimum(p, pmax)
            dm = data - next(g(p))
            neval += 1
            chi2 = (wgt * dm*dm).sum()
            if chi2 < 1.000001*chi2p or (p == p1).all():
                break
            # step with this lambda made things worse,
            # try bigger lambda for smaller steepest descent step
            lamda *= gain
            if lamda > lambdax:
                raise LevmarError("lambda grew beyond lambdax")

        conv = (chi2p - chi2)/(chi2p + chi2)
        if conv+conv <= tol:
            break
        # shrink lambda toward linearized least squares step
        lamda /= gain
        niter += 1
        if niter >= itmax:
            if not quiet:
                print("WARNING: levmar hit iteration limit %".format(niter))
            break

    chi20 /= ndof
    chi2 /= ndof
    pfull[pmask] = p
    pc = zeros((p.size, pfull.size), dtype=p.dtype)
    pc[:, pmask] = inv(alpha)
    pcov = zeros((pfull.size, pfull.size), dtype=p.dtype)
    pcov[pmask] = pc
    cfg.update(src='levmar', niter=niter, neval=neval, p0=p0, chi20=chi20,
               lamda=lamda)
    return ModelFit(f, pfull, pcov, chi2, ndof, isgenf, info=cfg)

# Default configuration options for levmar.  See levmar docstring.
# Perhaps should put matrix solver here as well.
levmar.config = dict(itmax=100, tol=1.e-7, lambda0=0.001, lambdax=1.e12,
                     gain=10., prel=1.e-6, pabs=1.e-9)


class ModelFit(object):
    """Best fit model of a parametrized family of models.

    If `model` is a ModelFit instance for a parametrized function
    ``f(p, a1, a2, ...)``, then ``model(a1, a2, ...)`` is the function
    with `p` set to the fit values, while ``model.p`` are the parameters
    themselves.

    Attributes
    ----------
    f : function or generator function
        The family of parameterized models.
    p : 1D ndarray
        The best fit parameters.
    pcov : 2D ndarray
        Covariances of `p`.  Note that these scale as ``errs**2``,the specified
        errors in the data.  If no errs were provided for the data,
        these covariances are relative; see `chi2pcov` below.
    chi2 : float
        The `chi2` per degree of freedom of the best fit to the data.
        Again, this scales as ``1/errs**2``.
    ndof : int
        The number of degrees of freedom in the fit (number of data points
        minus number of free parameters).
    isgenf : bool
        True if `f` is a generator function, else False.
    info : dict
        More detailed information about the fit, such as the function that
        did it and how it was configured, the number of iterations, the
        initial guess and corresponding `chi2`, etc.

    See Also
    --------
    levmar : non-linear least squares fitter
    regress : linear least squares fitter
    """
    def __init__(self, f, p, pcov=None, chi2=None, ndof=None, isgenf=None,
                 info=None, **kw):
        self.f, self.p = f, p
        if isgenf is None:
            isgenf = isgeneratorfunction(f)
        self.isgenf = isgenf
        if isgenf:
            def _f(*args, **kwargs):
                return next(f(p, *args, **kwargs))
        else:
            def _f(*args, **kwargs):
                return f(p, *args, **kwargs)
        self._f = _f
        self.pcov = pcov
        self.chi2 = chi2
        self.ndof = ndof
        self.isgenf = isgenf
        self.info = info
        # any other keywords simply become attributes
        self.__dict__.update(kw)

    def __call__(self, *args, **kwargs):
        """Model function, with parameter argument set to best fit values.

        Parameters
        ----------
        a1,a2,... : arbitrary
        k1=k1,k2=k2,... : arbitrary
            Non-parameter arguments of the parametrized function
            ``f(p, a1, a2, ..., k1=k1, k2=k2, ...)``

        Returns
        -------
        ndarray
            The function value ``f(p, a1, a2, ..., k1=k1, k2=k2, ...)``
            with `p` set to the best fit parameters.
        """
        return self._f(*args, **kwargs)

    @property
    def perr(self):
        """Standard deviations of parameters.

        The square root of the diagonal of `pcov`.
        """
        return sqrt(diag(self.pcov))

    @property
    def chi2pcov(self):
        """Covariance of parameters, errors estimated from quality of fit.

        Covariances estimated from the quality of this fit, assuming that
        no errs were provided to go with the data.  This amount to assuming
        that the `chi2` per degree of freedom of the fit is 1.0, and scaling
        `errs` retroactively to make that true.  Every statistical package
        provides this but warns against taking it too seriously.
        """
        return self.chi2 * self.pcov

    @property
    def chiperr(self):
        """Std deviations of params, errors estimated from quality of fit.

        The square root of the diagonal of `chi2pcov`.
        """
        return sqrt(diag(self.chi2pcov))

    def chi2prob(self, chi2=None):
        """Probability that chi2 per dof is greater than specified value.

        Parameters
        ----------
        chi2 : array_like, optional
            Chi square per degree of freedom.  If not specified, will use
            ``self.chi2``.  Note that this is `chi2` *per degree of freedom*.

        Returns
        -------
        prob : ndarray
            Probability that `chi2` of fit will be greater than given `chi2`.
        """
        if chi2 is None:
            chi2 = self.chi2
        chi2 = asfarray(chi2)
        hndof = 0.5 * self.ndof
        return gammaincc(hndof, hndof*chi2)


def numerical_partials(f, p, f0=None, pmin=None, pmax=None, prel=1.e-6,
                       pabs=1.e-9, pmask=None, args=(), kwargs=None):
    """Compute partial derivatives of f(p) wrt p by finite differences."""
    if kwargs is None:
        kwargs = {}
    f0, p = f(p) if f0 is None else asfarray(f0), asfarray(p)
    dp = zeros_like(p)
    prel, pabs = prel + dp, pabs + dp
    dp = maximum(prel*absolute(p), pabs)
    pfull = p.copy()
    if pmask is not None:
        p, dp = p[pmask], dp[pmask]
    else:
        pmask = ones(p.shape, dtype=bool)
    # Assume that pmin <= p <= pmax, but check p+dp.
    if pmax is not None:
        mask = p+dp > pmax
        dp[mask] *= -1
        if mask.any():
            if pmin is not None and (p+dp < pmin).any():
                raise ValueError("pmin and pmax too close together")
    dfdp = []
    for dp, p1 in zip(dp, p+diag(dp)):
        if not dp:
            raise ValueError("zero step size, check prel and pabs")
        pfull[pmask] = p1
        dfdp.append((f(pfull, *args, **kwargs) - f0)/dp)
    return array(dfdp)


class LevmarError(ValueError):
    """Levenberg-Marquardt algorithm lambda runaway, a kind of ValueError."""
    pass
