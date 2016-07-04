# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Interface for proper use of ColorBrewer qualitative color sets."""

from matplotlib import rc
try:
    from matplotlib import cycler
    cyc_name = 'prop_cycle'
except ImportError:
    # matplotlib 1.4 and below
    cycler = lambda x, y: y
    cyc_name = 'color_cycle'


class CBQ(object):
    """Container for ColorBrewer qualitative color sets.

    These are good choices for line color cycles, in addition to maps
    meant to distinguish twelve or fewer different regions.  Only the
    first three set2 and dark2 colors and the first four paired colors
    are colorblind safe.  Mostly these sets are distinguished by hue,
    with the colors in a set similar lightness and saturation::

        set1, pastel1            9 colors
        set2, pastel2, dark2     8 colors
        set3                    12 colors
        accent                   8 colors
        paired                  12 colors

        set1 darkest, most saturated
        dark2 next darkest, less saturated
        accent 4 lighter colors followed by 4 darker colors
        set2 more uniform about same darkness as lighter accent colors
        set3 intermediate between darker and more pastel sets
        pastel1 more variable lightness, more saturated than pastel2
        pastel2 lightest, least saturated
        paired[0::2] are pastel corresponding to darker paired[1::2]
           pastels are about at set3 lightness, darks about at set1

    The yellow color in set1, pastel1, set3, accent, and paired is not
    visible against white.  All other colors contrast with both white
    and black.  The following modified sets omit the yellow color::

        set1ny, pastel1ny      8 colors
        set3ny                11 colors
        accentny               7 colors

    Do not use the final pair in paired, that is use only paired[0:10],
    to avoid the yellow (which is paired[10]).
    """
    set1 = ('#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
            '#FFFF33', '#A65628', '#F781BF', '#999999')
    pastel1 = ('#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6',
               '#FFFFCC', '#E5D8BD', '#FDDAEC', '#F2F2F2')
    set2 = ('#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854',
            '#FFD92F', '#E5C494', '#B3B3B3')
    pastel2 = ('#B3E2CD', '#FDCDAC', '#CBD5E8', '#F4CAE4', '#E6F5C9',
               '#FFF2AE', '#F1E2CC', '#CCCCCC')
    dark2 = ('#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E',
             '#E6AB02', '#A6761D', '#666666')
    set3 = ('#8DD3C7', '#FFFFB3', '#BEBADA', '#FB8072', '#80B1D3', '#FDB462',
            '#B3DE69', '#FCCDE5', '#D9D9D9', '#BC80BD', '#CCEBC5', '#FFED6F')
    accent = ('#7FC97F', '#BEAED4', '#FDC086', '#FFFF99', '#386CB0',
              '#F0027F', '#BF5B17', '#666666')
    paired = ('#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C',
              '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A', '#FFFF99', '#B15928')

    # The yellows in set1, pastel1, set3, and accent are not visible against
    # a white background.  Remove them.
    # Reorder to roughly match matplotlib default bgrcmyk cycle.
    set1ny = ('#377EB8', '#4DAF4A', '#E41A1C', '#999999', '#984EA3',
              '#FF7F00', '#A65628', '#F781BF')
    pastel1ny = ('#B3CDE3', '#CCEBC5', '#FBB4AE', '#F2F2F2', '#DECBE4',
                 '#FED9A6', '#E5D8BD', '#FDDAEC')
    set3ny = ('#80B1D3', '#B3DE69', '#FB8072', '#8DD3C7', '#BC80BD', '#FFED6F',
              '#D9D9D9', '#FCCDE5', '#BEBADA', '#FDB462', '#CCEBC5')
    accentny = ('#386CB0', '#7FC97F', '#F0027F', '#BF5B17', '#BEAED4',
                '#FDC086', '#666666')

    @classmethod
    def set_color_cycle(cls, name, axes=None):
        """Set the color cycle for future plot commands.

        Parameters
        ----------
        name : str
            One of the color set names, see help(CBQ).
        axes : Axes, optional
            If not provided, sets the default color cycle in the rcParams
            axes section.  Otherwise, sets the color cycle for the given
            axes.
        """
        rc('axes', **{cyc_name: cycler('color', getattr(cls, name))})
