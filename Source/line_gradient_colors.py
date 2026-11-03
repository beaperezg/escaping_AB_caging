from typing import Optional
import numpy as np
import matplotlib.colors as colors
from matplotlib.legend_handler import HandlerTuple
import matplotlib
from scipy.interpolate import CubicSpline
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt




def color_line_lc(x: np.ndarray, y: np.ndarray, c: np.ndarray, ax: Optional[plt.axis] = None,
                  vmin: Optional[float] = None, vmax: Optional[float] = None, 
                  n_interpolate: Optional[int] = 1, **kwargs) -> LineCollection:
    """
    Plot a line with function y(x). Each segment is plotted with a color defined by the provided colormap
    and a magnitude defined by the numpy array c. If no cmap is defined, the default 'viridis' will be used.
    The cmap limits can be modified with the vmin and vmax optional parameters. If not provided the limits are
    given by the minimum and maximum value of the array c.
    If a more smooth line is desired a CubicSpline interpolation can be used to artificially increase the number
    of points in the curve, obtaining not only a smooth paht, but also a smooth transition between colors. The
    total number of points to draw is len(x) x n_interpolate.
    Optional arguents can be passed to the LineCollection instante in keyword arguments, such as: line_width,
    zorder, antialiased, ... ()
    
    WARNING: If interpolation is used, the x or y must be strictly in ascending order. If not, CubicSpline will
    raise an error.
    
    Parameters
    ----------
    x: np.ndarray
        x-coordinate
    y: np.ndarray
        y-coordinate
    c: np.ndarray
        Values for the colormap
    ax: Optional(plt.axis, default=None)
        Matpltolib axis where plot the data
    vmin: Optional(float, default=None)
        Minimum value for the normalization of the cmap. If not provided, the min(c) will be used
    vmax: Optional(float, default=None)
        Maximum value for the normalization of the cmap. If not provided, the max(c) will be used
    n_interpolate: Optional(int, default=1)
        Factor increase for the line interpolation
    **kwargs:
        Forwarded to matplotlib.collections
    
    Return
    ------
    lc: LineCollection
        Final result for the segments with gradient colors
        
    """
    ax = plt.gca() if ax is None else ax  # Use current ax if not provided

    if vmin is None:
        vmin = np.min(c)
    if vmax is None:
        vmax = np.max(c)

    if n_interpolate > 1:
        try:
            y_int = CubicSpline(x, y)
            c_int = CubicSpline(x, c)
            x = np.linspace(np.min(x), np.max(x), len(x) * n_interpolate)
            y = y_int(x)
            c = c_int(x)
        except ValueError:  # When computing vertical plots, where y is sorted ascending
            x_int = CubicSpline(y, x)
            c_int = CubicSpline(y, c)
            y = np.linspace(np.min(y), np.max(y), len(y) * n_interpolate)
            x = x_int(y)
            c = c_int(y)

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, norm=norm, capstyle='round', joinstyle='bevel', **kwargs)

    lc.set_array(c)

    ax.add_collection(lc)
    ax.autoscale()

    return lc