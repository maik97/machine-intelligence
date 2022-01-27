import numpy as np
from scipy import interpolate


def get_ramp_interpolator(
        val_a: float,
        val_b: float,
        point_a: float,
        point_b: float,
        kind: str = 'linear',
        *args, **kwargs
) -> interpolate.interp1d:
    """
    Initialize a 1-D linear interpolation class, to interpolate a ramp. Points will be assigned as follows:

        x = [0.0, point_a, point_b, 1.0]
        y = [val_a,val_a, val_b, val_b]

    :param val_a:
        Corresponds as y to given x by point_a
    :param val_b:
        Corresponds as y to given x by point_b
    :param point_a:
        Sets ramp start point, value on x-axis between 0.0 and 1.0
    :param point_b:
        Sets ramp end point, value on x-axis between 0.0 and 1.0
    :param kind:
        Documentation taken from scipy:
            Specifies the kind of interpolation as a string or as an integer specifying the order of the spline
            interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline
            interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’ simply return the previous or
            next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating half-integers (e.g. 0.5, 1.5)
            in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
    :return:
        interpolate.interp1d object, that interpolates the ramp, when called
    """
    return interpolate.interp1d(
        x=np.array([0.0, point_a, point_b, 1.0]),
        y=np.array([val_a,val_a, val_b, val_b]),
        kind=kind,
        *args, **kwargs
    )
