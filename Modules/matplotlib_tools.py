import numpy
import scipy.stats
import matplotlib

def scatterplot_dense(axes, x, y, s=100, cmap="YlGnBu_r"):
    xy = numpy.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy, 0.03)(xy)
    idx = z.argsort()
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]
    scatter_plot = axes.scatter(x, y, c=z, s=s, edgecolor='', cmap=cmap)
    return scatter_plot
