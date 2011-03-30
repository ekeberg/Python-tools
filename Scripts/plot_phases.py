#! /usr/bin/python

import sys, pylab, spimage

log_flags = ['log']

def plot_phases(in_file,*arguments):
    flags = ['histogram','phases']
    plot_flag = 0
    log_flag = 0
    def no_log(x):
        return x

    fig = pylab.figure(1)
    ax = fig.add_subplot(111)
n
    try:
        img = spimage.sp_image_read(in_file,0)
    except:
        print "Error when reading %s.\n" % in_file
        return
    values = img.image.reshape(pylab.size(img.image))

    for flag in arguments:
        if flag in flags:
            plot_flag = flag
        elif flag in log_flags:
            log_flag = flag
        else:
            print "unknown flag %s" % flag

    if log_flag == 'log':
        log_function = pylab.log
    else:
        log_function = no_log

    if plot_flag == 'phases':
        hist = pylab.histogram(pylab.angle(values),bins=50)
        ax.plot((hist[1][:-1]+hist[1][1:])/2.0,log_function(hist[0]))
    elif plot_flag == 'histogram':
        hist = pylab.histogram2d(pylab.real(values),pylab.imag(values),bins=500)
        ax.imshow(log_function(hist[0]),extent=(hist[2][0],hist[2][-1],-hist[1][-1],-hist[1][0]),interpolation='nearest')
    else:
        ax.plot(pylab.real(values),pylab.imag(values),'.')
    return fig
    


if __name__ == "__main__":
    try:
        plot_phases(sys.argv[1],*sys.argv[2:])
        pylab.show()
    except:
        print """
Usage: plot_phases datafile [flags]

List of flags:
histogram
phases
"""
