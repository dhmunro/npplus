# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Numpy enhancements and alternatives.

Modules
-------
basic : basic APIs, many inspired by yorick
    a_, cat_ : broadcasting versions of hstack and concatenate
    span, spanl, max_, min_, abs_, atan : more flexible than numpy versions
    cum, zcen, pcen : rank preserving axis methods to supplement numpy.diff

pcwise : a decorator function, alternative to numpy.piecewise

pwpoly : multidimensional piecewise polynomial interpolation
"""
