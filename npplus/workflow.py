"""Interactive workflow convenience module.

Merely importing this adds its functions to the builtins module dictionary,
there is no need to access the module's own namespace.
Do not import this module in other modules; it is solely for interactive use,
or use in a PYTHONSTARTUP file.

Alternatively, cut and paste this code directly into a PYTHONSTARTUP file.

Functions
---------
reload : expose this, crucial for interactive debug workflow
execfile : easy to understand for non-expert interactive users
"""

__all__ = ['reload', 'execfile']

import sys

if sys.version_info >= (3,0):
    # make reload, execfile look like they do in python2
    # for interactive sessions -- do not use in scripts
    import builtins
    if sys.version_info >= (3,4):
        from importlib import reload
    else:
        from imp import reload
    builtins.__dict__['reload'] = reload
    def execfile(*args):
        """Python2 execfile -- avoid this, write modules and use import."""
        name, args = args[0], args[1:] if len(args)>1 else (globals(),)
        with open(name) as f:
            code = compile(f.read(), name, 'exec')
            eval("exec(code, *args)")  # evade syntax error in python2
    builtins.__dict__['execfile'] = execfile
