# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Interactive wrappers for interactive use of various pyplot APIs.

Functions like ``matplotlib.pyplot.plot(x,y)`` are usually invoked as
subroutine calls interactively, but inconveniently return an object
which prints at the terminal.  This module wraps such functions to
return None, so that an interactive terminal isn't half filled with
unwanted object outputs.  In the rare cases in which you need the
return value, you can easily use, e.g.- ``result = plt.plot(x,y)``.

Also provides the convenience functions `xylim` and `logxy`.

--------
"""

# require alternatives: axis
__all__ = ['annotate', 'axhline', 'axhspan', 'axvline', 'axvspan',
           'bar', 'barbs', 'barh', 'boxplot', 'broken_barh',
           'colorbar', 'contour', 'contourf', 'errorbar', 'eventplot',
           'figimage', 'figlegend', 'figtext', 'figure', 'fill',
           'fill_between', 'fill_betweenx', 'hexbin', 'hist',
           'hist2d', 'hlines', 'imshow', 'legend', 'pcolor',
           'pcolormesh', 'pie', 'plot', 'plot_date', 'quiver',
           'scatter', 'stem', 'step', 'streamplot', 'subplot',
           'suptitle', 'text', 'title', 'tricontour', 'tricontourf',
           'tripcolor', 'triplot', 'violinplot', 'vlines', 'xlabel',
           'xlim', 'ylabel', 'ylim']

from functools import wraps as _wraps
import matplotlib.pyplot as plt  # name plt cannot appear in __all__ list


def _iwrap(name):
    f = getattr(plt, name)

    @_wraps(f)
    def iwrapped(*args, **kwargs):
        f(*args, **kwargs)

    globals()[name] = iwrapped

for _ in __all__:
    try:
        _iwrap(_)
    except AttributeError:
        # Happens in two important cases:
        # 1. Old versions of matplotlib do not have violinplot.
        # 2. ReadTheDocs uses mock matplotlib.pyplot with no names.
        exec(_+'=NotImplemented')  # otherwise "from pltwraps import *" fails


def xylim(xmin=None, xmax=None, ymin=None, ymax=None):
    """Set xmin, xmax, ymin, ymax limits for the current axes, return None.

    Parameters
    ----------
    xmin,xmax,ymin,ymax : float, optional
        Axis limit or None to leave unchanged.

    See Also
    --------
    matplotlib.pyplot.axis : get or set limits for current axes

    Notes
    -----
    The `xylim` function avoids the need for passing a tuple to `axis`,
    and has no return value, to avoid printing a value.  It is strictly
    a convenience function for interactive use; use `axis` in scripts.
    """
    plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)


def logxy(islog=None, islogy=Ellipsis):
    """Alternative to xscale, yscale for setting log or linear axes.

    Parameters
    ----------
    islog : bool, optional
        True means to use log scale for axes, False means linear scale.
        None or omitted means leave axis scaling unchanged.
    islogy : bool, optional
        If islogy omitted, islog applies to both x and y axes.
        If islogy provided, islog applies only to the x axis and islogy
        applies to the y axis; values have the same meaning as for islog.
    """
    if islog is not None:
        plt.xscale('log' if islog else 'linear')
    if islogy is Ellipsis:
        islogy = islog
    if islogy is not None:
        plt.yscale('log' if islogy else 'linear')


__all__ += ['xylim', 'logxy']  # defined here
