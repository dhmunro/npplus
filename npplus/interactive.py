# Copyright (c) 2016, David H. Munro
# All rights reserved.
# This is Open Source software, released under the BSD 2-clause license,
# see http://opensource.org/licenses/BSD-2-Clause for details.
"""Import most names in npplus for interactive use.

from npplus.interactive import *

This module should be imported only from PYTHONSTARTUP, except for
similar interactive.py modules in other packages.
"""

# sys and os too common to have to import them
import sys, os

# duplicate the pylab numpy imports for interactive use
import numpy as np
import numpy.ma as ma
from numpy import *
from numpy.fft import *
from numpy.random import *
from numpy.linalg import *
# fix clobbered datetime and bytes (np.random.bytes)
# note: modern numpy does not clobber datetime, but this is harmless
import datetime
bytes = __builtins__['bytes']

# force sane SIGFPE error handling
np.seterr(divide='raise', over='raise', invalid='raise')

# give interpreted access to npplus modules
from .yorlike import *
from .pwpoly import *
from .pcwise import *

# implement deprecated execfile for python3
if sys.version_info >= (3,0):
    if sys.version_info >= (3,4):
        from importlib import reload
    else:
        from imp import reload
    def execfile(*args):
        """python2 execfile -- avoid this, write modules and use import."""
        name, args = args[0], args[1:] if len(args)>1 else (globals(),)
        with open(name) as f:
            code = compile(f.read(), name, 'exec')
            eval("exec(code, *args)")  # evade syntax error in python2

# implement extended reload for development reloadx-pdb-edit cycle
def reloadx(module):
    """reload(module); from module import *

    # You may want to softlink 'ln -s /path/to/module_or_package .'
    # in your user site-packages directory (python -m site --user-site).
    # See site module in python standard library documentation.
    import module    # begin by importing your module or package
    reloadx(module)  # same as from module import *
    # test your module
    import pdb       # use pdb.pm(), pdb.run("test...") to debug
    # edit module.py (or package/module.py)
    reloadx(module)  # reload module and re-import its symbols
    # be sure to recreate any objects constructed from modified classes
    # loop debug, edit, reloadx
    """
    reload(module)
    import __main__
    for nm in vars(module):
        setattr(__main__, nm, getattr(module, nm))
