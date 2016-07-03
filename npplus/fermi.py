"""Fermi-Dirac integral and inverse of orders -1/2, 1/2, 3/2, 5/2
Antia, H.M., Aph.J 84, 101-108 (1993)
"""

# fermi integral related to polylogarithm:  F_j(x) = -Li_j+1(-exp(x))

__all__ = ['fdm12', 'fd12', 'fd32', 'fd52']

from functools import wraps
from numpy import asfarray, zeros_like, exp, polyval, sqrt


# The algorithm for the various orders is almost identical,
# differing only in the rational function coefficients and
# the power of x in the x>2 case.
# Use a decorator to apply the code template.
def _fd_integ(anum, aden, bnum, bden):
    def fdwrapper(f):
        @wraps(f)
        def fdtemplate(x):
            x = asfarray(x)
            shape, x = x.shape, x.ravel()
            mask = x <= 2.
            y = zeros_like(x)
            z = x[mask]
            if z.size:
                z = exp(z)
                y[mask] = (z * polyval(anum, z) / polyval(aden, z))
            mask = ~mask
            x = x[mask]
            if x.size:
                z = 1. / (x * x)
                x = sqrt(x) * f(x)  # note only use of original f
                y[mask] = (x * polyval(bnum, z) / polyval(bden, z))
            return y.reshape(shape)
        return fdtemplate
    return fdwrapper


@_fd_integ((1.0, 1.98276889924768e3, 1.14980998186874e5, 1.83696370756153e6,
            1.14587609192151e7, 3.16743385304962e7, 3.88148302324068e7,
            1.71446374704454e7),
           (4.35061725080755e2, 3.13595854332114e4, 6.13709569333207e5,
            4.81648022267831e6, 1.77657027846367e7, 3.26070130734158e7,
            2.87386436731785e7, 9.67282587452899e6),
           (1.0, 2.98435207466372e0, -7.45519953763928e-1,
            -8.52408612877447e-1, -1.60926102124442e-1, -1.12295393687006e-2,
            -3.69976170193942e-4, -6.64932238528105e-6, -6.84738791621745e-8,
            -4.44467627042232e-10, -1.58654991146236e-12,
            -4.46620341924942e-15),
           (4.16485970495288e-1, 1.86795964993052e0, -4.99759250374148e-1,
            -4.78770844009440e-1, -8.34904593067194e-2, -5.69764436880529e-3,
            -1.86432212187088e-4, -3.33919612678907e-6, -3.43299431079845e-8,
            -2.22564376956228e-10, -7.94193282071464e-13,
            -2.23310170962369e-15))
def fdm12(x):
    """Fermi-Dirac integral of order -1/2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        ``integral[0 to inf]{ dt * t**(-0.5) / (exp(t-x)+1) }``
        accurate to about 1e-12.
    """
    return 1.0


@_fd_integ((1.0, 7.77238678539648e2, 4.16031909245777e4, 6.42493233715640e5,
            3.93536421893014e6, 1.07608632249013e7, 1.30964880355883e7,
            5.75834152995465e6),
           (9.02129136642157e1, 8.17922106644547e3, 1.95155948326832e5,
            1.83167424554505e6, 7.95192647756086e6, 1.69288134856160e7,
            1.70750501625775e7, 6.49759261942269e6),
           (1.0, 1.91247528779676e0, 1.08037861921488e0, 2.48653216266227e-1,
            2.60768398973913e-2, 1.32212995937796e-3, 3.40679845803144e-5,
            4.69233883900644e-7, 3.76794942277806e-9, 1.64429113030738e-11,
            4.85378381173415e-14),
           (-2.14562434782759e-2, 2.01311836975930e-1, 1.34981244060549e0,
            1.16434871200131e0, 3.24095226486468e-1, 3.66887808002874e-2,
            1.92040136756592e-3, 5.02360015186394e-5, 6.96888634549649e-7,
            5.62152894375277e-9, 2.45745452167585e-11, 7.28067571760518e-14))
def fd12(x):
    """Fermi-Dirac integral of order -1/2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        ``integral[0 to inf]{ dt * t**0.5 / (exp(t-x)+1) }``
        accurate to about 1e-12.
    """
    return x


@_fd_integ((1.0, 9.90562948053193e1, 2.21876607796460e3, 1.77294861572005e4,
            5.95275291210962e4, 8.55472308218786e4, 4.32326386604283e4),
           (1.99507945223266e-2, 5.05580641737527e0, 2.20853967067789e2,
            3.20803912586318e3, 1.95942074576400e4, 5.50859144223638e4,
            7.01022511904373e4, 3.25218725353467e4),
           (1.0, 9.02250179334496e-1, 2.78383256609605e-1,
            3.85682997219346e-2, 2.62988766922117e-3, 9.12915407846722e-5,
            1.63598843752050e-6, 1.62974620742993e-8, 8.60096863656367e-11,
            2.80452693148553e-13),
           (2.34829436438087e-3, -2.15540156936373e-2, 4.70252591891375e-1,
            3.14236143831882e-1, 6.39899717779153e-2, 5.31999109566385e-3,
            2.04569943213216e-4, 3.84703231868724e-6, 3.94452010378723e-8,
            2.10699282897576e-10, 7.01131732871184e-13))
def fd32(x):
    """Fermi-Dirac integral of order -1/2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        ``integral[0 to inf]{ dt * t**1.5 / (exp(t-x)+1) }``
        accurate to about 1e-12.
    """
    return x*x


@_fd_integ((1.0, 1.02589947781696e2, 2.44325236813275e3, 2.10427138842443e4,
            7.67255995316812e4, 1.20132462801652e5, 6.61606300631656e4),
           (3.31482978240026e-3, 1.16951072617142e0, 6.35483623268093e1,
            1.10886130159658e3, 7.97584657659364e3, 2.60117136841197e4,
            3.79076097261066e4, 1.99078071053871e4),
           (1.0, 3.42040216997894e0, 3.18831203950106e0, 1.24415366126179e0,
            2.39564845938301e-1, 2.32779790773633e-2, 1.14008027400645e-3,
            2.77981736000034e-5, 3.54323824923987e-7, 2.31618876821567e-9,
            8.42667076131315e-12),
           (2.27326643192516e-2, 3.70866321410385e-1, 5.31886045222680e-1,
            2.27132567866839e-1, 3.99937801931919e-2, 2.81111224925648e-3,
            8.09451165406274e-5, 1.12919616415947e-6, 7.68215783076936e-9,
            2.94933476646033e-11))
def fd52(x):
    """Fermi-Dirac integral of order -1/2.

    Parameters
    ----------
    x : array_like

    Returns
    -------
    ndarray
        ``integral[0 to inf]{ dt * t**2.5 / (exp(t-x)+1) }``
        accurate to about 1e-12.
    """
    return x*x*x
