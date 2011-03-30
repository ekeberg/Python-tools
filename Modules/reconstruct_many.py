#from pylab import *
#import pylab
import os
import re
import Queue

class Reconstructer:
    def __init__(self, confFile, intsName, baseDir, nRecs, confDict):
        """Calls the repeat reconstruction script to do many reconstructions. Made to be used together with the MultipleReconstructions class. Takes the configuration file to use, the intensities file, a directory in which to put the result, the number of reconstructions to perform and a dictionary containing changes to make to the conf file (as "name" : "value")."""
        self.intsName = intsName
        self.baseDir = baseDir
        self.nRecs = nRecs
        self.confDict = confDict
        self.recName = os.path.basename(self.intsName)[:-3]
        self.recDir = '%s/%s' % (self.baseDir, self.recName)
        if not os.path.isdir(self.recDir):
            os.mkdir(self.recDir)
        os.chdir(self.recDir)
        self.confFile = self.editConf(confFile)
    def editConf(self, confFile):
        f = file(confFile, 'r')
        lines = f.readlines()
        f.close()
        index = [i for i in range(len(lines)) if re.search('intensities_file',lines[i])][0]
        #index = lines.index('  intensities_file =;\n')
        lines[index] = '  intensities_file = \"%s\";\n' % self.intsName
        for k in self.confDict.keys():
            index = [i for i in range(len(lines)) if re.search('%s =' % k,lines[i])][0]
            reResult = re.search('^(.*)=', lines[index])
            if reResult:
                s = reResult.groups()[0]
                lines[index] = '%s = %s;\n' % (s, self.confDict[k])
        newFile = '%s/uwrapc.conf' % (self.recDir) 
        f = file(newFile, 'wp')
        f.writelines(lines)
        f.close()
        return newFile
    def start(self):
        """Starts the reconstruction"""
        os.system('/usr/local/bin/repeat_reconstruction.pl %d' % self.nRecs)

class MultipleReconstructions:
    """Performs multiple reconstructions of multiple files using the Reconstructer class."""
    def __init__(self, confFile, intsDir, baseDir, nRecs, confDict):
        self.confFile = confFile
        self.intsDir = intsDir
        self.baseDir = baseDir
        self.nRecs = nRecs
        self.confDict = confDict
        self.files = self.getFileList()
    def getFileList(self):
        files = os.listdir(self.intsDir)
        files = ['%s/%s' % (self.intsDir, f) for f in files if re.search('\.h5$', f)]
        files.sort()
        return files
    def start(self):
        for f in self.files:
            print 'Reconstruct %s' % f
            Reconstructer(self.confFile, f, self.baseDir, self.nRecs, confDict).start()

if __name__ == "__main__":
    confDict = {'max_iterations' : '5000',
                'work_directory' : '\".\"',
                'output_period' : '1000',
                'log_output_period' : '500'}

    MultipleReconstructions('/data/LCLS2011/r0138/all/uwrapc.conf',
                            '/home/ekeberg/Work/python/mimi_finder/preprocess/processed_for_reconstruction',
                            '/data/LCLS2011/r0138/all',
                            30, confDict).start()


    
