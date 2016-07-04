NpPlus - numpy and pyplot Enhancements
======================================

See complete documentation at http://npplus.readthedocs.io.

Highlights
----------

1. A high performance piecewise polynomial class PwPoly supporting more
   methods than the scipy PPoly class, including a root finder and
   arithmetic operations.
2. Simplified spline and polyline interpolating and fitting functions
   fully integrated with PwPoly, using just the solve_banded lapack
   function instead of fitpack.
3. Simplified interfaces for linear and non-linear least squares fitting.
4. A function decorator to replace the clumsy numpy.piecewise function.
5. Array building functions which broadcast their arguments before
   concatenating them.
6. Finite difference axis methods supplementing the existing diff
   and cumsum functions.
7. Elementwise minimum and maximum functions taking any number of arguments.
8. Ensure presence of matplotlib 2.0 colormaps viridis, etc., and provide
   an interface for using the qualitative colorbrewer color sets for line
   color cycles.
9. Provide a simple presentation-quality matplotlib style.
10. Provide wrappers for pyplot functions like plot that return unwanted
    objects, making them better for interactive use.
11. Provide a module you can import from PYTHONSTARTUP similar to pylab,
    which fills interactive namespace with numpy and npplus, as well as
    turning on the pyplot interactive plotting mode.
