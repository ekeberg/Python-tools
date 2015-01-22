#!/bin/env python
import sys
import pylab
from optparse import OptionParser

PLOTS = ["it", "ereal", "efourier", "fc_fo", "sup_size", "beta", "threshold", "algorithm", "d_ereal", "d_sup_size", "blur_radius", "int_cum_fluct", "object_area", "phase_relation_error", "correlation_with_solution", "phase_blur_radius", "delta_rho", "iterations_s"]

NAMES = [r'Outerloop iterations', r'$E_{real}$', r'$E_{fourier}$', r'$\left<\frac{F_c}{F_0}\right>$', r'Support Size', r'$\beta$', r'Threshold', r'Algorithm', r'$\Delta E_{real}$', r'$\Delta$ Support Size', r'Blur Radius', r'Int Cum Fluct', r'Object Area', r'Phase Relation Error', r'Correlation With Solution', r'Phase Blur Radius', r'$\Delta \rho$', r'Iterations/s']

def plot_log(log_file, plot_types):
    try:
        data = pylab.loadtxt(log_file,skiprows=43)
    except:
        IOError("Can not read %s" % log_file)

    indices = []

    if len(plot_types) == 0:
        plot_types = ["ereal", "efourier"]

    for plot_type in plot_types:
        if plot_type in PLOTS:
            indices.append(PLOTS.index(plot_type)+1)

    for i in indices:
        pylab.plot(data[:,0],data[:,i],label=NAMES[i-1])

    pylab.legend()

if __name__ == "__main__":
    import tools

    parser = OptionParser(usage="%prog [-p plot1 -p plot2 ...] <logfile.log>")
    parser.add_option("-p", action="append", type="choice", dest="plots", help="Specify a plot to plot.",
                      choices=("it", "ereal", "efourier", "fc_fo", "sup_size", "beta", "threshold", "algorithm", "d_ereal", "d_sup_size", "blur_radius", "int_cum_fluct", "object_area", "phase_relation_error", "correlation_with_solution", "phase_blur_radius", "delta_rho", "iterations_s"), default=[])
    (options, args) = parser.parse_args()
    
    tools.remove_duplicates(options.plots)

    if len(options.plots) == 0:
        options.plots.append("ereal")
        options.plots.append("efourier")
    
    if len(args) == 0:
        raise IOError("Need to provide a log file") 

    plot_log(args[0], options.plots)
    pylab.show()
    
    
