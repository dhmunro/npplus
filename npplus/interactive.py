"""Import most names in npplus for interactive use.

import npplus.interactive

This module should be imported only from PYTHONSTARTUP, except for
similar interactive.py modules in other packages, which should use
the npplus.interactive.import_all_to_main function.
"""

import __main__
def import_all_to_main(*modules):
    """same as typing 'from module import *' in interactive session"""
    for m in modules:
        names = getattr(m, '__all__', None)
        if names is None:
            names = [k for k in m.__dict__.keys() if not k.startswith('_')]
        for nm in names:
            setattr(__main__, nm, getattr(m, nm))

# sys and os too common to have to import them
import sys, os
__main__.sys, __main__.os = sys, os

# duplicate the pylab numpy imports for interactive use
import numpy as np
import numpy.ma as ma
__main__.np, __main__.ma = np, ma
import_all_to_main(np, np.fft, np.random, np.linalg)
# fix clobbered datetime and bytes (np.random.bytes)
# note: modern numpy does not clobber datetime, but this is harmless
import datetime
__main__.datetime, __main__.bytes = datetime, __builtins__['bytes']
# force sane SIGFPE error handling
np.seterr(divide='raise', over='raise', invalid='raise')

# give interpreted access to npplus modules
from . import yorlike, pwpoly, pcwise
import_all_to_main(yorlike, pwpoly, pcwise)

# implement deprecated execfile for python3
if sys.version_info >= (3,0) and not hasattr(__main__, 'execfile'):
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
    __main__.execfile, __main__.reload = execfile, reload

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
    import_all_to_main(reload(module))
__main__.reloadx = reloadx
