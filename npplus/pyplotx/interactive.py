"""Set up for interactive plotting with matplotlib.pylab.

from npplus.pyplotx.interactive import *
# uncomment to get inward ticks and box (default outward, no box)
#style_npp(box=True)
"""

# add npplus.interactive
from ..interactive import *

# recommended scipy main packages
#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
# pylab = all of above imports, plus most commands into interactive namespace
from pylab import *

# wrap plot(), etc. so they return None for use at terminal
from .pltwraps import *

# fix matplotlib defaults for better interactive experience, add style_npp
from .mpldefaults import *

# turn on matplotlib.pyplot interactive mode
plt.ion()
