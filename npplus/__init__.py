# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Numpy and pyplot enhancements and alternatives.

Modules
-------
basic :
    Basic APIs, many inspired by yorick.
pwpoly :
    Piecewise polynomial interpolation and fitting.
lsqfit :
    Linear and non-linear least squares fits of data to models.
pcwise :
    A decorator function, alternative to numpy.piecewise.
solveper :
    Periodic banded matrix variants of scipy.linalg.solve_banded.
fermi :
    Fermi-Dirac integrals and inverses.
pyplotx :
    Package providing mpl-2 colormaps viridis, etc., and nice plot style.
interactive :
    Make npplus and numpy available (for PYTHONSTARTUP files).  Also provides
    `reloadx` function to simplify interactive debugging workflow.
pyplotx.interactive :
    Interactive+pyplot plus quiet plotting function wrappers and ``ion()``.
"""

from .basic import *
from .pwpoly import *
from .pcwise import *
from .lsqfit import *
