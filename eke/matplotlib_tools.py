import numpy as _numpy
import scipy as _scipy
import scipy.stats
import matplotlib as _matplotlib
import matplotlib.pyplot
import matplotlib.widgets
import matplotlib.colors
import itertools as _itertools
import abc as _abc


def scatterplot_dense(x, y, s=100, cmap="YlGnBu_r", ax=None):
    """Plot a scatterplot with colorcode for the density."""
    if ax == None:
        ax = _matplotlib.pyplot.gca()
    xy = _numpy.vstack([x, y])
    #z = _scipy.stats.gaussian_kde(xy, 0.03)(xy)
    sigma = _numpy.mean((x.max() - x.min(), y.max() - y.min())) * s / 10000
    z = _scipy.stats.gaussian_kde(xy, sigma)(xy)
    idx = z.argsort()
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]
    scatter_plot = ax.scatter(x_sorted, y_sorted, c=z_sorted, s=s, edgecolor='', cmap=cmap)
    return scatter_plot


class BaseBrowser(object):
    __metaclass__ = _abc.ABCMeta
    def __init__(self, data, layout=(1, 1)):
        self._data = data
        self._layout = layout

        self._number_of_plots = len(data)
        self._plots_per_page = _numpy.product(layout)
        self._number_of_pages = (self._number_of_plots-1) / self._plots_per_page + 1
        self._current_page = 0

        with _matplotlib.rc_context({"toolbar": "None"}):
            self._fig = _matplotlib.pyplot.figure("Browser")
            self._fig.clear()
            #_matplotlib.pyplot.show()
        self._fig.set_facecolor("white")
        self._fig.subplots_adjust(left=0., bottom=0.1, right=1., top=0.95, wspace=0., hspace=0.05)

        self._axes = [self._fig.add_subplot(layout[0], layout[1], 1+i0*layout[1]+i1) for i0, i1 in _itertools.product(xrange(layout[0]), xrange(layout[1]))]

        ax_prev = self._fig.add_axes([0.1, 0.01, 0.3, 0.09])
        ax_next = self._fig.add_axes([0.6, 0.01, 0.3, 0.09])

        self._button_prev = _matplotlib.widgets.Button(ax_prev, "Previous")
        self._button_next = _matplotlib.widgets.Button(ax_next, "Next")
        self._button_prev.on_clicked(self._prev)
        self._button_next.on_clicked(self._next)

        self._page_text = self._fig.text(0.5, 0.05, "", va="center", ha="center")
        self._update_page_string()
        self._first_plot()

    def _local_to_global_index(self, local_index):
        return self._current_page*self._plots_per_page+local_index

    def _prev(self, event):
        if self._current_page == 0:
            return
        self._current_page -= 1
        self._plot()
        self._update_page_string()

    def _next(self, event):
        if self._current_page == self._number_of_pages-1:
            return
        self._current_page += 1
        self._plot()
        self._update_page_string()

    def _update_page_string(self):
        self._page_text.set_text("{current_low} - {current_high} of {max_index}".format(
            current_low=self._current_page*self._plots_per_page,
            current_high=min((self._current_page+1)*self._plots_per_page-1, self._number_of_plots),
            max_index=self._number_of_plots))

    @_abc.abstractmethod
    def _first_plot(self):
        pass

    @_abc.abstractmethod
    def _plot(self):
        pass

class ImageBrowser(BaseBrowser):
    def __init__(self, data, layout, log=False):
        if log:
            self._norm = _matplotlib.colors.LogNorm()
        else:
            self._norm = _matplotlib.colors.NoNorm()
        super(ImageBrowser, self).__init__(data, layout)

    def _first_plot(self):
        self._plot_array = []
        for local_index in xrange(self._plots_per_page):
            self._plot_array.append(self._axes[local_index].imshow(self._data[self._local_to_global_index(local_index)], norm=self._norm))
            self._axes[local_index].axis("off")
            self._fig.canvas.draw()

    def _plot(self):
        for local_index in xrange(self._plots_per_page):
            if self._local_to_global_index(local_index) >= self._number_of_plots:
                self._plot_array[local_index].set_data(_numpy.zeros(self._data[0].shape))
            else:
                self._plot_array[local_index].set_data(self._data[self._local_to_global_index(local_index)])
                self._fig.canvas.draw()

class PlotBrowser(BaseBrowser):
    def __init__(self, data, layout):
        super(PlotBrowser, self).__init__(data, layout)

    def _first_plot(self):
        self._plot_array = []
        for local_index in xrange(self._plots_per_page):
            self._plot_array.append(self._axes[local_index].plot(self._data[self._local_to_global_index(local_index)]))
            self._fig.canvas.draw()

    def _plot(self):
        for local_index in xrange(self._plots_per_page):
            self._plot_array[local_index][0].set_data(_numpy.arange(len(self._data[0])), self._data[self._local_to_global_index(local_index)])
            self._fig.canvas.draw()

class BaseScatterplotPick(object):
    __metaclass__ = _abc.ABCMeta
    def __init__(self, x, y, data):
        self._data = data
        self._fig = _matplotlib.pyplot.figure("ScatterPick")
        self._fig.clear()
        self.scatter_ax = self._fig.add_subplot(121)
        self.data_ax = self._fig.add_subplot(122)

        #self.scatter_plot = self.scatter_ax.scatter(x, y, picker=tolerance=10)
        self.scatter_plot = self.scatter_ax.plot(x, y, "o", color="black", picker=10)
        self._first_plot()
        #self._cid_pick = self._fig.canvas.mpl_connect("pick_event", self._on_pick)
        self._cid_pick = self._fig.canvas.callbacks.connect("pick_event", self._on_pick)

    def _on_pick(self, event):
        if event.mouseevent.inaxes != self.scatter_ax:
            return
        index = event.ind[0]
        self._plot(index)

    @_abc.abstractmethod
    def _first_plot(self):
        pass

    @_abc.abstractmethod
    def _plot(self, index):
        pass


class ImageScatterplotPick(BaseScatterplotPick):
    def __init__(self, x, y, data, log=False):
        if log:
            self._norm = _matplotlib.colors.LogNorm(clip=True)
        else:
            self._norm = _matplotlib.colors.NoNorm()

        super(ImageScatterplotPick, self).__init__(x, y, data)

    def _first_plot(self):
        self.data_plot = self.data_ax.imshow(self._data[0], norm=self._norm)
        self._fig.canvas.draw()

    def _plot(self, index):
        self.data_plot.set_data(self._data[index])
        self._fig.canvas.draw()

class PlotScatterplotPick(BaseScatterplotPick):
    def __init__(self, x, y, data):
        super(PlotScatterplotPick, self).__init__(x, y, data)

    def _first_plot(self):
        self.data_plot = self.data_ax.plot(self._data[0])
        self._fig.canvas.draw()

    def _plot(self, index):
        self.data_plot[0].set_data(self._data[index])
        self._fig.canvas.draw()
