"""Import most names in npplus for interactive use.

from npplus.full import *
"""

from .yorlike import spanl, max_, min_, abs_, atan, cum, zcen, pcen
from .pcwise import pcwise
from .pwpoly import PwPoly

__all__ = ['spanl', 'max_', 'min_', 'abs_', 'atan', 'cum', 'zcen', 'pcen',
           'pcwise', 'PwPoly']
