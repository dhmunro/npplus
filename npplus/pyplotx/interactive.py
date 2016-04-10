"""Set up for interactive plotting with matplotlib.pylab.

from mplplus.interactive import *
"""

# recommended scipy main packages
#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
# pylab = all of above imports, plus most commands into interactive namespace
from pylab import *

# add npplus.interactive
from ..interactive import *

# fix matplotlib defaults for better interactive experience
from .mpltweak import *

# turn on matplotlib.pyplot interactive mode
plt.ion()
