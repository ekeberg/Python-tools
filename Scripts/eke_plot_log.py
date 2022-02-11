#!/usr/bin/env python
import numpy
import matplotlib
import matplotlib.pyplot
import argparse


PLOTS = ["it", "ereal", "efourier", "fc_fo", "sup_size", "beta", "threshold",
         "algorithm", "d_ereal", "d_sup_size", "blur_radius", "int_cum_fluct",
         "object_area", "phase_relation_error", "correlation_with_solution",
         "phase_blur_radius", "delta_rho", "iterations_s"]
NAMES = [r'Outerloop iterations', r'$E_{real}$', r'$E_{fourier}$',
         r'$\left<\frac{F_c}{F_0}\right>$', r'Support Size', r'$\beta$',
         r'Threshold', r'Algorithm', r'$\Delta E_{real}$',
         r'$\Delta$ Support Size', r'Blur Radius', r'Int Cum Fluct',
         r'Object Area', r'Phase Relation Error', r'Correlation With Solution',
         r'Phase Blur Radius', r'$\Delta \rho$', r'Iterations/s']


def plot_log(log_file, plot_types):
    try:
        data = numpy.loadtxt(log_file, skiprows=43)
    except IOError:
        IOError(f"Can not read {log_file}")

    indices = []

    if len(plot_types) == 0:
        plot_types = ["ereal", "efourier"]

    for plot_type in plot_types:
        if plot_type in PLOTS:
            indices.append(PLOTS.index(plot_type)+1)

    for i in indices:
        matplotlib.pyplot.plot(data[:, 0], data[:, i], label=NAMES[i-1])

    matplotlib.pyplot.legend()


if __name__ == "__main__":
    from eke import tools

    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-p" "--plot", action="append",
                        choices=PLOTS,
                        help="Specify a plot to plot.")

    args = parser.parse_args()

    tools.remove_duplicates(args.plot)

    if len(args.plot) == 0:
        args.plot.append("ereal")
        args.plot.append("efourier")

    plot_log(args.file, args.plot)
    matplotlib.pyplot.show()
