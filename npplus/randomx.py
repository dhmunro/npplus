# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Generate random random points and rotations in and on circles and spheres.

These functions are consistent with the numpy.random random(size)
interface, in order to be compatible with the numerous other random
distributions provided in numpy.random.

--------

"""

from numpy.random import random
from numpy import pi, sin, cos, sqrt
from numpy import concatenate, newaxis, roll, transpose, prod

# intrinsic timings 100000 samples (us):
#   x*x, x**2       265
#   x*y, x/y        357, 400
#   x*y*z           720
#   sqrt            518
#   x**3          12300
#   exp(3*log(x)) 11400 !!
#   random(1e5)    2070
#   sin, cos       4700, 4080
#   exp, expm1     4860, 4580
#   log, log1p     6210, 4250
#   standard_normal  8000 (3.9 * random)

#   oncircle      17700   (8.6 * random)
#   incircle      15100   (7.3 * random)
#   onsphere      25000  (12.1 * random)
#   insphere      27000  (13.0 * random)
#   rotation3     52800


def incircle(size=None):
    """Return uniform random points inside 2D unit circle.

    Parameters
    ----------
    size : int or tuple of int
        Leading axes of returned points.  Final axis always length 2.

    Returns
    -------
    xy : ndarray
        The random points inside the unit circle, trailing dimension 2.
    """
    if size is None:
        size = ()
    else:
        try:
            size = tuple(size)
        except TypeError:
            size = (size,)
    n = int(prod(size))
    if n < 330:
        # For small n, interpreted overhead dominates.  Using sin and cos
        # results in fewer interpreted instructions than rejection method.
        # Compiled code should never use this algorithm.
        t, z = random((2,) + size + (1,))
        t *= 2. * pi
        return sqrt(z) * concatenate((cos(t), sin(t)), axis=-1)
        # Beats this slightly:
        # xy = standard_normal(size + (2,))
        # return xy * expm1(-0.5 * (xy*xy).sum(axis=-1, keepdims=True))
    # For large n, higher intrinsic cost of sin and cos compared to
    # rejection method dominates, and it is worth taking a few more
    # interpreted instructions to benefit from the superior algorithm.
    nmore = n
    p = []
    fac = 4./pi  # 1/prob random point in unit circle
    while nmore > 0:  # Odds of needing another pass < 0.0001.
        m = int((nmore + 5.*sqrt(nmore))*fac)
        q = 2.*random((m, 2)) - 1.
        q = q[(q * q).sum(axis=-1) < 1., :]
        p.append(q)
        nmore -= len(q)
    return concatenate(p)[:n].reshape(size + (2,))


def oncircle(size=None):
    """Return uniform random points on 2D unit circle.

    Parameters
    ----------
    size : int or tuple of int
        Leading axes of returned points.  Final axis always length 2.

    Returns
    -------
    xy : ndarray
        The random points on the unit circle, trailing dimension 2.
    """
    if size is None:
        size = ()
    else:
        try:
            size = tuple(size)
        except TypeError:
            size = (size,)
    # This beats normalizing incircle for all sizes, even though that
    # should be the superior algorithm for compiled code.
    theta = 2.*pi * random(size + (1,))
    return concatenate((cos(theta), sin(theta)), axis=-1)


def insphere(size=None):
    """Return uniform random points inside 3D unit sphere.

    Parameters
    ----------
    size : int or tuple of int
        Leading axes of returnined points.  Final axis always length 3.

    Returns
    -------
    xyz : ndarray
        The random points inside the unit sphere, trailing dimension 3.
    """
    if size is None:
        size = ()
    else:
        try:
            size = tuple(size)
        except TypeError:
            size = (size,)
    n = int(prod(size))
    if n < 70:
        # For small n, interpreted overhead dominates.  Using sin and cos
        # results in fewer interpreted instructions than rejection method.
        # Compiled code should never use this algorithm.
        mu, phi, z = random((3,) + size + (1,))
        mu = 2.*mu - 1.
        phi *= 2. * pi
        s = sqrt(1. - mu)
        return z**(1./3.) * concatenate((s*cos(phi), s*sin(phi), mu), axis=-1)
        # Beats this:
        # p = onsphere(size)
        # return p * random(p.shape[:-1] + (1,)) ** (1./3.)
    # For large n, higher intrinsic cost of sin and cos compared to
    # rejection method dominates, and it is worth taking a few more
    # interpreted instructions to benefit from the superior algorithm.
    nmore = n
    p = []
    fac = 6./pi  # 1/prob random point in unit sphere
    while nmore > 0:
        m = int((nmore + 5.*sqrt(nmore))*fac)  # 99.9+% chance of nmore
        q = 2.*random((m, 3)) - 1.
        q = q[(q * q).sum(axis=-1) < 1., :]
        nmore -= len(q)
        p.append(q)
    return concatenate(p)[:n].reshape(size + (3,))


def onsphere(size=None):
    """Return uniform random points on 3D unit sphere.

    Parameters
    ----------
    size : int or tuple of int
        Leading axes of returnined points.  Final axis always length 3.

    Returns
    -------
    xyz : ndarray
        The random points on the unit sphere, trailing dimension 3.
    """
    xy = oncircle(size)
    z = 2.*random(xy.shape[:-1] + (1,)) - 1.
    xy *= sqrt(1. - z*z)
    return concatenate((xy, z), axis=-1)


def rotation3(size=None):
    """Return a collection of 3x3 random rotation matrices.

    Parameters
    ----------
    size : int or tuple of int, optional
        The number or leading dimensions of the 3x3 matrices returned.

    Returns
    -------
    rotmat : ndarray
        A single 3x3 rotation matrix (that is, three orthogonal unit vectors
        in right-hand order) if `size` not given.  Otherwise the dimensions
        specified by `size` are the leading dimensions of the collection of
        3x3 rotation matrices.

    Notes
    -----

    By "random", we mean that these matrices transform any single unit
    vector to a collection of unit vectors uniformly distributed over
    the surface of a sphere.

    Uses the Arvo algorithm from Graphics Gems III.
    See http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    and http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.53.1357&rep=rep1&type=pdf
    and https://en.wikipedia.org/wiki/Rotation_matrix#Uniform_random_rotation_matrices

    """
    if size is None:
        size = ()
    else:
        try:
            size = tuple(size)
        except TypeError:
            size = (size,)
    theta, phi, z = 2. * random((3, 1) + size)
    theta *= pi  # Initial rotation angle about z-axis.
    phi *= pi  # Angle in xy plane for tilt of z-axis.
    # Magnitude of tilt is random variable z.
    r = sqrt(z)
    v = concatenate((r*sin(phi), r*cos(phi), sqrt(2.-z)))
    st, ct = sin(theta), cos(theta)
    s = concatenate((v[0]*ct - v[1]*st, v[0]*st + v[1]*ct))
    m = v[:, newaxis].repeat(3, axis=1)
    m[:, :2] *= s
    m[0, :2] -= concatenate((ct, st))
    m[1, :2] += concatenate((st, -ct))
    m[:2, 2] *= v[2]
    m[2, 2] = 1. - z  # Equals v[2]*v[2] - 1.
    if m.ndim > 2:
        m = transpose(m, roll(range(m.ndim), -2)).copy()
    return m

# literal transcription of Graphics Gems III rand_rotation.c for testing
# from numpy import ones, zeros, array
# def croutine(x):
#     theta, phi, z = 2.*pi*x[0], 2.*pi*x[1], 2.*x[2]
#     r = sqrt(z)
#     vx, vy, vz = r*sin(phi), r*cos(phi), sqrt(2.-z)
#     st, ct = sin(theta), cos(theta)
#     sx, sy = vx*ct - vy*st, vx*st + vy*ct
#     m = zeros((3,3,)+x.shape[1:])
#     m[0, 0] = vx*sx - ct
#     m[0, 1] = vx*sy - st
#     m[0, 2] = vx*vz
#     m[1, 0] = vy*sx + st
#     m[1, 1] = vy*sy - ct
#     m[1, 2] = vy*vz
#     m[2, 0] = vz*sx
#     m[2, 1] = vz*sy
#     m[2, 2] = 1. - z
#     return m
