import numpy as _numpy
import scipy as _scipy
import scipy.stats
import matplotlib as _matplotlib
import matplotlib.pyplot
import matplotlib.widgets
import matplotlib.colors
import itertools as _itertools
import abc as _abc
from six import with_metaclass as _with_metaclass


def scatterplot_dense(x, y, s=100, cmap="YlGnBu_r", ax=None):
    """Plot a scatterplot with colorcode for the density."""
    if ax is None:
        ax = _matplotlib.pyplot.gca()
    xy = _numpy.vstack([x, y])
    sigma = _numpy.mean((x.max() - x.min(), y.max() - y.min())) * s / 10000
    z = _scipy.stats.gaussian_kde(xy, sigma)(xy)
    idx = z.argsort()
    x_sorted = x[idx]
    y_sorted = y[idx]
    z_sorted = z[idx]
    scatter_plot = ax.scatter(x_sorted, y_sorted,
                              c=z_sorted, s=s, edgecolor='none', cmap=cmap)
    return scatter_plot


class BaseBrowser(_with_metaclass(_abc.ABCMeta)):
    def __init__(self, data, layout=(1, 1)):
        self._data = data
        self._layout = layout

        self._number_of_plots = len(data)
        self._plots_per_page = _numpy.product(layout)
        self._number_of_pages = ((self._number_of_plots-1) //
                                 self._plots_per_page + 1)
        self._current_page = 0

        with _matplotlib.rc_context({"toolbar": "None"}):
            self._fig = _matplotlib.pyplot.figure("Browser")
            self._fig.clear()
        self._fig.set_facecolor("white")
        self._fig.subplots_adjust(left=0., bottom=0.1, right=1., top=0.95,
                                  wspace=0., hspace=0.05)

        self._axes = [self._fig.add_subplot(layout[0], layout[1],
                                            1+i0*layout[1]+i1)
                      for i0, i1 in _itertools.product(range(layout[0]),
                                                       range(layout[1]))]

        ax_prev = self._fig.add_axes([0.1, 0.01, 0.3, 0.09])
        ax_next = self._fig.add_axes([0.6, 0.01, 0.3, 0.09])

        self._button_prev = _matplotlib.widgets.Button(ax_prev, "Previous")
        self._button_next = _matplotlib.widgets.Button(ax_next, "Next")
        self._button_prev.on_clicked(self._prev)
        self._button_next.on_clicked(self._next)

        self._page_text = self._fig.text(0.5, 0.05, "",
                                         va="center", ha="center")
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
        current_low = self._current_page*self._plots_per_page
        current_high = min((self._current_page+1)*self._plots_per_page-1,
                           self._number_of_plots)
        max_index = self._number_of_plots
        self._page_text.set_text(f"{current_low} - {current_high} of "
                                 f"{max_index}")

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
        for local_index in range(self._plots_per_page):
            this_plot = self._axes[local_index].imshow(
                self._data[self._local_to_global_index(local_index)],
                norm=self._norm
            )
            self._plot_array.append(this_plot)
            self._axes[local_index].axis("off")
            self._fig.canvas.draw()

    def _plot(self):
        for local_index in range(self._plots_per_page):
            if (
                    self._local_to_global_index(local_index) >=
                    self._number_of_plots
            ):
                self._plot_array[local_index].set_data(
                    _numpy.zeros(self._data[0].shape))
            else:
                self._plot_array[local_index].set_data(
                    self._data[self._local_to_global_index(local_index)])
                self._fig.canvas.draw()


class PlotBrowser(BaseBrowser):
    def __init__(self, data, layout):
        super(PlotBrowser, self).__init__(data, layout)

    def _first_plot(self):
        self._plot_array = []
        for local_index in range(self._plots_per_page):
            self._plot_array.append(
                self._axes[local_index].plot(
                    self._data[self._local_to_global_index(local_index)]
                )
            )
            self._fig.canvas.draw()

    def _plot(self):
        for local_index in range(self._plots_per_page):
            self._plot_array[local_index][0].set_data(
                _numpy.arange(len(self._data[0])),
                self._data[self._local_to_global_index(local_index)]
            )
            self._fig.canvas.draw()


class BaseScatterplotPick(_with_metaclass(_abc.ABCMeta)):
    def __init__(self, x, y, data):
        self._data = data
        self._fig = _matplotlib.pyplot.figure("ScatterPick")
        self._fig.clear()
        self.scatter_ax = self._fig.add_subplot(121)
        self.data_ax = self._fig.add_subplot(122)

        self.scatter_plot = self.scatter_ax.plot(x, y, "o",
                                                 color="black", picker=10)
        self._first_plot()
        self._cid_pick = self._fig.canvas.callbacks.connect(
            "pick_event", self._on_pick)

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


def complex_plot(array, vmin=None, vmax=None, log=False):
    if vmin is None:
        vmin = abs(array).min()
    if vmax is None:
        vmax = abs(array).max()
    h = _numpy.mod(_numpy.angle(array), 2.*_numpy.pi)/(2.*_numpy.pi)
    if log:
        s = 0.85 * _numpy.ones_like(_numpy.log(abs(array)))
        v = _numpy.log(abs(array)) / _numpy.log(vmax)
    else:
        s = 0.85 * _numpy.ones_like(abs(array))
        v = (abs(array) - vmin) / (vmax - vmin)
    return _matplotlib.colors.hsv_to_rgb(_numpy.dstack((h, s, v)))


def plot_diffraction_pattern(pattern, wavelength, detector_distance,
                             pixel_size, ax=None):
    from eke import diffraction
    if ax is None:
        ax = _matplotlib.pyplot.gca()
    ewald = diffraction.ewald_coordinates([s+1 for s in pattern.shape],
                                          wavelength, detector_distance,
                                          pixel_size)
    return ax.pcolor(ewald[0], ewald[1], pattern)


def complex_to_rgb(data, vmax=None):
    absmax = vmax or abs(data).max()
    hsv = _numpy.zeros(data.shape + (3, ), dtype="float")
    hsv[..., 0] = _numpy.angle(data) / (2 * _numpy.pi) % 1
    hsv[..., 1] = 1
    hsv[..., 2] = _numpy.clip(abs(data) / absmax, 0, 1)
    rgb = _matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def imshow_array(fig, data, titles=None, **kwargs):
    nimages = len(data)
    row_col_ratio = fig.get_figheight() / fig.get_figwidth()

    ncols = int(round(_numpy.sqrt(nimages/row_col_ratio)))
    nrows = (nimages-1) // ncols + 1

    for i in range(nimages):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.imshow(data[i], **kwargs)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[i])
    fig.tight_layout()
