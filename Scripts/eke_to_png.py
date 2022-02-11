#!/usr/bin/env python
"""
This program converts all .h5 files in the current
directory to .png using the HAWK program
image_to_png
"""
import os
import re
import spimage
import argparse


class PlotSetup(object):
    scales = {"Phase": spimage.SpColormapPhase,
              "Jet": spimage.SpColormapJet,
              "Gray": spimage.SpColormapGrayScale}

    def __init__(self):
        self.color = self.scales["Jet"]
        self.shift = False
        self.log = 0
        self.mask = False

    def set_color(self, color_string):
        try:
            self.color = self.scales[color_string]
        except KeyError:
            pass

    def set_log(self, log_bool):
        self.log = log_bool*spimage.SpColormapLogScale

    def get_color(self):
        return self.color | self.log

    def set_shift(self, shift_bool):
        self.shift = shift_bool

    def get_shift(self):
        return self.shift

    def set_mask(self, mask_bool):
        self.mask = mask_bool

    def get_mask(self):
        return self.mask


def read_files(directory):
    all_files = os.listdir(directory)
    files = ['%s/%s' % (directory, f) for f in all_files
             if re.search('.h5$', f)]
    files.sort()
    return files


def evaluate_arguments(arguments):
    if len(arguments) <= 0:
        print("""
    This program converts all h5 files in the curren directory to png.
    Usage:  python_script_to_png [colorscale]

    Colorscales:
    Jet
    Gray
    PosNeg
    InvertedPosNeg
    Phase
    InvertedPhase
    Log (can be combined with the others)
    Shift (can be combined with the others)
    Support

    """)
        return
    elif not (isinstance(arguments, list) or isinstance(arguments, tuple)):
        print("function to_png takes must have a list or string input")
        return

    log_flag = 0
    shift_flag = 0
    support_flag = 0
    color = 16

    for flag in arguments:
        if flag == 'PosNeg':
            color = 8192
        elif flag == 'InvertedPosNeg':
            color = 16384
        elif flag == 'Phase':
            color = 256
        elif flag == 'InvertedPhase':
            color = 4096
        elif flag == 'Jet':
            color = 16
        elif flag == 'Gray':
            color = 1
        elif flag == 'Log':
            log_flag = 1
        elif flag == 'Shift':
            shift_flag = 1
        elif flag == 'Support':
            support_flag = 1
        else:
            print("unknown flag %s" % flag)

    if log_flag == 1:
        color += 128
    return color, shift_flag, support_flag


def get_shift_function(bool):
    if bool:
        def shift_function(img):
            ret = spimage.sp_image_shift(img)
            spimage.sp_image_free(img)
            return ret
    else:
        def shift_function(img):
            return img
    return shift_function


def get_support_function(bool):
    if bool:
        def support_function(img):
            spimage.sp_image_mask_to_image(img, img)
    else:
        def support_function(img):
            pass
    return support_function


def to_png_parallel(input_files, output_dir, plot_setup):
    import multiprocessing
    import queue

    class Worker(multiprocessing.Process):
        def __init__(self, working_queue, out_dir, process_function, color):
            multiprocessing.Process.__init__(self)
            self.working_queue = working_queue
            self.out_dir = out_dir
            self.process_function = process_function
            self.color = color

        def process(self, f):
            img = spimage.sp_image_read(f, 0)
            img = self.process_function(img)
            output_file = self.out_dir + "/" + f[:-2] + "png"
            spimage.sp_image_write(img, output_file, self.color)
            spimage.sp_image_free(img)

        def run(self):
            while not self.working_queue.empty():
                try:
                    print(f"{self.name} get new, approx "
                          f"{self.working_queue.qsize()} left")
                except NotImplementedError:
                    pass
                try:
                    f = self.working_queue.get()
                except queue.Empty:
                    break
                self.process(f)
            print("%s done" % self.name)

    def split_files(files, n):
        from pylab import split, array
        rest = len(files) % n
        if rest:
            super_list = split(array(files)[:-rest], len(files)//n)
            super_list.append(files[-rest:])
        else:
            super_list = split(array(files), len(files)//n)
        return super_list

    def run_threads(nThreads):
        shift_function = get_shift_function(plot_setup.get_shift())
        support_function = get_support_function(plot_setup.get_mask())

        def process_function(img):
            support_function(img)
            return shift_function(img)

        working_queue = multiprocessing.Queue()
        for f in input_files:
            working_queue.put(f)

        for i in range(nThreads):
            Worker(working_queue, output_dir, process_function,
                   plot_setup.get_color()).start()

    run_threads(multiprocessing.cpu_count())


def to_png(input_dir, output_dir, plot_setup):
    files = read_files(input_dir)

    support_function = get_support_function(plot_setup.get_mask())

    for f in files:
        img = spimage.sp_image_read(f, 0)
        support_function(img)
        img = spimage.sp_image_shift(img)
        output_file = output_dir + "/" + f[:-2] + "png"
        spimage.sp_image_write(img, output_file, plot_setup.get_color())
        spimage.sp_image_free(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", default=".",
                        help="Input directory. Default is .")
    parser.add_argument("outdir", default=".",
                        help="Output directory. Default is .")
    parser.add_argument("-c", "--colorscale", default="jet",
                        help="Colorscale")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Log scale")
    parser.add_argument("-s", "--shift", action="store_true",
                        help="Shift image")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Plot mask")
    args = parser.parse_args()

    plot_setup = PlotSetup()
    plot_setup.set_color(args.colorscale)
    print(args.log)
    plot_setup.set_log(args.log)
    plot_setup.set_shift(args.shift)
    plot_setup.set_mask(args.mask)

    files = read_files(args.indir)

    to_png_parallel(files, args.outdir, plot_setup)
