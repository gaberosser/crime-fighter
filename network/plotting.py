__author__ = 'gabriel'
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.path as mpath

def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=None,
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    norm = norm or plt.Normalize(z.min(), z.max())

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def network_point_coverage(net, dx=None, include_nodes=True):
    '''
    Produce a series of semi-regularly-spaced points on the supplied network.
    :param net: Network
    :param dx: Optional spacing between points, otherwise this is automatically selected
    :param include_nodes: If True, points are added at node locations too
    :return: (i) (E x N(i) x 2) array of Cartesian points, E is the # edges, N(i) is the # points on edge i
             (ii) E x N(i) array of (closest_edge, dist_along) tuples)
    TODO: switch to using NetworkPoint objects
    '''

    # small delta to avoid errors
    eps = 1e-6

    ## temp set dx with a constant
    xy = []
    cd = []
    dx = dx or 1
    for e in net.edges(data=True):
        this_xy = []
        this_cd = []
        interp_lengths = np.arange(eps, e[2]['length'] - eps, dx)
        interp_lengths = np.concatenate((interp_lengths, [e[2]['length'] - eps]))
        # interpolate along linestring
        ls = e[2]['linestring']
        interp_pts = [ls.interpolate(t) for t in interp_lengths]

        for i in range(interp_lengths.size):
            this_xy.append((interp_pts[i].x, interp_pts[i].y))
            d = {
                e[2]['orientation_neg']: interp_lengths[i],
                e[2]['orientation_pos']: e[2]['length'] - interp_lengths[i],
            }
            this_cd.append((e, d))
        xy.append(this_xy)
        cd.append(this_cd)

    return xy, cd


if __name__ == "__main__":
    N = 100
    np.random.seed(101)
    x = np.random.rand(N).cumsum()
    y = np.random.rand(N).cumsum()
    fig, ax = plt.subplots()

    path = mpath.Path(np.column_stack([x, y]))
    verts = path.interpolated(steps=3).vertices
    x, y = verts[:, 0], verts[:, 1]
    z = np.linspace(0, 10, len(x))
    colorline(x, y, z, cmap=plt.get_cmap('jet'), linewidth=2)

    plt.axis('auto')

    plt.show()