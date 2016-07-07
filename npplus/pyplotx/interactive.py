# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Set up for interactive plotting with matplotlib.pylab.

Put this in your PYTHONSTARTUP file::

    from npplus.pyplotx.interactive import *
    # uncomment to get default npplus style (outward ticks, no box)
    #style_npp()
    # uncomment to get inward ticks and box
    #style_npp(box=True)

You do not need to import ``npplus.interactive`` if you import
``npplus.pyplotx.interactive``, it will be done automatically.

--------
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
try:
    plt.ion()
except NameError:
    # ReadTheDocs mock import needs this
    pass
