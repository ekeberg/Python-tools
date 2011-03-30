#from pylab import *
#import pylab
import os
import re
import shutil
import Queue
import multiprocessing

class PRTF:
    """Calculate prtf after a multiple reconstruction. Puts the result in a directory named prtf in baseDir."""
    def __init__(self, baseDir, prefix):
        self.baseDir = baseDir
        self.prefix = prefix
        self.prtfDir = '%s/prtf' % baseDir
        dirs = os.listdir(self.baseDir)
        self.dirs = ["%s/%s" % (self.baseDir, d) for d in dirs if re.search('^[0-9]{6}$',d)]
        self.dirs.sort()
        self.nRecs = len(self.dirs)
    def getFinalName(self):
        files = os.listdir(self.dirs[0])
        real_files = [f for f in files if re.search('^real_space-[0-9]{7}\.h5$',f)]
        real_files.sort()
        return real_files[-1]
    def start(self):
        if self.nRecs == 0:
            return 0
        self.finalName = self.getFinalName()
        files = ['%s/%s' % (d,self.finalName) for d in self.dirs]
        file_list = " ".join(files)
        if not os.path.isdir(self.prtfDir):
            os.mkdir(self.prtfDir)
        os.chdir(self.prtfDir)
        os.system('prtf %s %s' % (self.prefix, file_list))
        return 1
    def copyFinal(name,dest):
        if name:
            shutil.copy("%s/%s-%s" % (self.prtfDir, self.prefix, name), dest)
        else:
            shutil.copy("%s/%s" % (self.prtfDir, self.prefix), dest)

class MultiplePRTF(multiprocessing.Process):
    """Do a prtf in every directory in baseDir with more reconstructions than nLim inside."""
    def __init__(self, baseDir, nLim, prefix, finalDest, working_queue):
        multiprocessing.Process.__init__(self)
        self.baseDir = baseDir
        self.nLim = nLim
        self.prefix = prefix
        self.finalDest = finalDest
        self.working_queue = working_queue
    def run(self):
        #print self.dirs
        while not self.working_queue.empty():
            try:
                d = self.working_queue.get()
            except Queue.Empty:
                break
            print "%s : %s" % (self.name, d)
            print "%s : %d left" % (self.name, self.working_queue.qsize())
            p = PRTF(d,self.prefix)
            if p.nRecs >= self.nLim:
                print "%s do" % self.name
                p.start()
                if os.path.isfile('%s/prtf/%s-avg_image.h5' % (d, self.prefix)):
                    shutil.copy('%s/prtf/%s-avg_image.h5' % (d, self.prefix),
                                '%s/%s.h5' % (self.finalDest, os.path.basename(d)))
                print "%s done" % self.name
            else:
                print "%s dont" % self.name

def start_prtfs(baseDir, nLim, prefix, outDir, cpu_count):
    dirs = ['%s/%s' % (baseDir, d) for d in os.listdir(baseDir) if os.path.isdir('%s/%s' % (baseDir,d)) and not os.path.isdir('%s/%s/prtf' % (baseDir, d))]
    print '%s/%s/prtf' % (baseDir, d)
    dirs.sort()
    working_queue = multiprocessing.Queue()
    for d in dirs:
        working_queue.put(d)

    for i in range(cpu_count):
        MultiplePRTF(baseDir, nLim, prefix, outDir, working_queue).start()
        


if __name__ == '__main__':
    start_prtfs('/data/LCLS2011/r0138/all', 30, 'mimi',
                '/data/LCLS2011/r0138/all/average_images', 3)
    # m = MultiplePRTF('/data/LCLS2011/r0138/all',30,'mimi')
    # m.start()
    # m.copyFinalReal('/data/LCLS2011/r0138/all/average_images')
    
