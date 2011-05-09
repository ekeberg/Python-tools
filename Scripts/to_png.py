
"""
This program converts all .h5 files in the current
directory to .png using the HAWK program
image_to_png
"""
import os, re, sys, spimage
from guppy import hpy

def read_files():
    l = os.listdir('.')
    files = [f for f in l if  re.search('.h5$',f)]
    return files

def evaluate_arguments(arguments):
    if len(arguments) <= 0:
        print """
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

    """
        return
    elif not (isinstance(arguments,list) or isinstance(arguments,tuple)):
        print "function to_png takes must have a list or string input"
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
            print "unknown flag %s" % flag

    if log_flag == 1:
        color += 128
    return color,shift_flag,support_flag
    
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
            spimage.sp_image_mask_to_image(img,img)
    else:
        def support_function(img):
            pass
    return support_function

def to_png_parallel(*arguments):
    import multiprocessing
    import Queue

    class Worker(multiprocessing.Process):
        def __init__(self, working_queue, process_function, color):
            multiprocessing.Process.__init__(self)
            self.working_queue = working_queue
            self.process_function = process_function
            self.color = color
        def process(self, f):
            img = spimage.sp_image_read(f,0)
            img = self.process_function(img)
            spimage.sp_image_write(img,f[:-2]+"png",self.color)
            spimage.sp_image_free(img)
        def run(self):
            while not self.working_queue.empty():
                try:
                    print "%s get new, approx %d left" % (self.name,self.working_queue.qsize())
                except NotImplementedError:
                    pass
                try:
                    #f = self.working_queue.get_nowait()
                    f = self.working_queue.get()
                except Queue.Empty:
                    break
                self.process(f)
                # h = hpy()
                # print "%s:\n" % self.name
                # print h.heap()
            print "%s done" % self.name

    def split_files(files,n):
        from pylab import split, array
        rest = len(files)%n
        if rest:
            super_list = split(array(files)[:-rest],len(files)/n)
            super_list.append(files[-rest:])
        else:
            super_list = split(array(files),len(files)/n)
        return super_list

    def run_threads(arguments,nThreads):
        color,shift_flag,support_flag = evaluate_arguments(arguments)
        shift_function = get_shift_function(shift_flag)
        support_function = get_support_function(support_flag)
        def process_function(img):
            support_function(img)
            return shift_function(img)
        
        files = read_files()
        #super_list = split_files(files,10)

        working_queue = multiprocessing.Queue()
        # for job in super_list:
        #     working_queue.put(job)
        for f in files:
            working_queue.put(f)

        for i in range(nThreads):
            Worker(working_queue, process_function, color).start()

    run_threads(arguments,multiprocessing.cpu_count())

def to_png(*arguments):
    color,shift_flag,support_flag = evaluate_arguments(arguments)
    files = read_files()

    shift_function = get_shift_function(shift_flag)
    support_function = get_support_function(support_flag)

    for f in files:
        img = spimage.sp_image_read(f,0)
        support_function(img)
        img = sp_image_shift(img)
        spimage.sp_image_write(img,f[:-2]+"png",color)
        spimage.sp_image_free(img)

if __name__ == "__main__":
    to_png_parallel(*sys.argv[1:])

