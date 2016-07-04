# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Provide some matplotlib default settings and repackaging.

Exposes magma, inferno, plasma, and viridis colormaps, and CBQ object
encapsulating the ColorBrewer qualitative palettes for use as color cycles.

Provides ``style_npp()`` function to set nicer default matplotlib style:

1. Ensure matplotlib 2.0 colormaps present and viridis default.
2. Simple CBQ interface for ColorBrewer color sets for linestyle cycling.
   Make default color cycle based on ColorBrewer Set1 (minus yellow).
3. Fix savefig bounding box default to not clip title and labels.
4. Make imshow not change viewport aspect by default.
5. Change to a more yorick-like default plot style, with much larger
   axis labels, outward ticks, and a much subtler background.
6. Define title function that works with outward ticks.

--------
"""

# The plot style could have been put in a matplotlib stylelib file
# to take advantage of the mpl.style.use() interface, but that requires
# users to have special matplotlib knowledge to install it.

__all__ = ['magma', 'inferno', 'plasma', 'viridis', 'CBQ', 'title',
           'style_npp']

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from .cbqcolors import CBQ
try:
    from matplotlib.pyplot import magma, inferno, plasma, viridis
except ImportError:
    from .mpl2cmaps import magma, inferno, plasma, viridis


def style_npp(box=None):
    """Set rcParams for nicer style than matplotlib defaults.

    Parameters
    ----------
    box : bool, optional
        If set, inward ticks with box drawn around the plot.  By default,
        the style is less traditional with outward ticks that can never
        obscure data and no box around the plot.

    Notes
    -----
    Other features are:
    *  Matplotlib 2.0 viridis colormap becomes default.
    *  Color cycle for plot lines derived from ColorBrewer Set1.
    *  Heavier lines and larger axis numbering.
    *  Images displayed with imshow do not change aspect ratio of axes.
    *  The savefig command set up to not clip axis labels or title.
    *  The background of figure and axes differ more subtlely on screen.
    """

    # Make viridis the default colormap.
    rc('image', cmap='viridis')

    # color cycle for plot() lines
    CBQ.set_color_cycle('set1ny')  # 'dark2' a good second choice

    # Without this savefig clips title and label text.
    # The downside is, this may break some animation backends, but we assume
    # an interactive graphics window backend.
    rc('savefig', bbox='tight')    # bbox_inches as savefig keyword
    rc('savefig', pad_inches=0.02)

    # Make imshow leave axes aspect alone and rescale image pixels.
    # This assumes your images are data with possibly very non-square pixels.
    # The matplotlib default assumes they are photographs which you want to
    # view with square pixels.
    rc('image', aspect='auto')

    rc('figure', facecolor='0.9')
    rc('figure', edgecolor='white')
    rc('axes', facecolor='white')
    if not box:
        rc('axes', edgecolor='0.8')
        rc('axes', linewidth=1.0)
    else:
        rc('axes', edgecolor='k')
        rc('axes', linewidth=1.4)

    rc('font', size=14.0)
    rc('axes', titlesize='x-large')
    rc('axes', labelsize='large')
    rc('xtick', labelsize='large')
    rc('ytick', labelsize='large')

    rc('lines', linewidth=3)
    rc('lines', markersize=10)

    # more yorick-like axes
    rc('xtick.major', size=6)
    rc('xtick.major', width=1.4)
    rc('ytick.major', size=6)
    rc('ytick.major', width=1.4)
    if not box:
        rc('xtick', direction='out')
        rc('ytick', direction='out')
    else:
        rc('xtick', direction='in')
        rc('ytick', direction='in')


# need to use title(text, y=1.03) with outward ticks
def title(s, *args, **kwargs):
    y = kwargs.get('y')
    ret = kwargs.pop('ret', None)
    if y is None and rcParams['xtick.direction'] == 'out':
        kwargs = dict(y=1.03, **kwargs)
    plt.title(s, *args, **kwargs)
    # use plt.title if you need the return value, like pltwraps
title.__doc__ = plt.title.__doc__
