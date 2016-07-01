# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Numpy and pyplot enhancements and alternatives.

Modules
-------
interactive : make npplus available (for PYTHONSTARTUP files)
pyplotx.interactive : interactive+pylab, including plwraps
pyplotx.plwraps : wrappers for pyplot APIs to improve usability

basic : basic APIs, many inspired by yorick
    a_, cat_ : broadcasting versions of hstack and concatenate
    span, spanl, max_, min_, abs_, atan : more flexible than numpy versions
    cum, zcen, pcen : rank preserving axis methods to supplement numpy.diff

pwpoly : piecewise polynomial interpolation and fitting
lsqfit : linear and non-linear least squares fits of data to models
pcwise : a decorator function, alternative to numpy.piecewise
solveper : periodic banded matrix solvers based on scipy solve_banded
fermi : Fermi-Dirac integrals and inverses of orders -1/2, 1/2, 3/2, 5/2
"""

from .basic import *
from .pwpoly import *
from .pcwise import *
from .lsqfit import *
