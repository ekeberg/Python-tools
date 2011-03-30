#! /usr/bin/python

import sys
import pylab

def plot_log(log_file, *arguments):
    try:
        data = pylab.loadtxt(log_file,skiprows=43)
    except:
        print "Error: Can not read %s" % log_file
        return

    plots = ["it", "ereal", "efourier", "fc_fo", "sup_size", "beta", "threshold", "algorithm", "d_ereal", "d_sup_size", "blur_radius", "int_cum_fluct", "object_area", "phase_relation_error", "correlation_with_solution", "phase_blur_radius", "delta_rho", "iterations_s"]

    names = [r'Outerloop iterations', r'$E_{real}$', r'E_{fourier}$', r'$\left<\frac{F_c}{F_0}\right>$', r'Support Size', r'$\beta$', r'Threshold', r'Algorithm', r'$\Delta E_{real}$', r'$\Delta$ Support Size', r'Blur Radius', r'Int Cum Fluct', r'Object Area', r'Phase Relation Error', r'Correlation With Solution', r'Phase Blur Radius', r'$\Delta \rho$', r'Iterations/s']


    indices = []

    if len(arguments) > 0:
        for arg in arguments:
            if arg in plots:
                indices.append(plots.index(arg)+1)
            elif arg == "all":
                indices=range(1,len(plots)+1)
                break
    else:
        indices = [2,3] #by default we plot real and fourier error

    for i in indices:
        pylab.plot(data[:,0],data[:,i],label=names[i-1])

    pylab.legend()

if __name__ == "__main__":
    try:
        plot_log(sys.argv[1],*sys.argv[2:])
        pylab.show()
    except:
        print """
Usage: python_script_plot_log <uwrapc.log> [plots]

Available plots are:
"it", "ereal", "efourier", "fc_fo", "sup_size", "beta", "threshold", "algorithm", "d_ereal", "d_sup_size", "blur_radius", "int_cum_fluct", "object_area", "phase_relation_error", "correlation_with_solution", "phase_blur_radius", "delta_rho", "iterations_s"

If none is specified ereal and efourier are plotted.
If "all" is specified all plots are plotted
"""
        

