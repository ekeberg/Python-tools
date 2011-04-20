from pylab import *

def get_h5_in_dir(path):
    "Returns a list of all the h5 files in a directory"
    import os
    import re
    l = os.listdir(path)
    files = ["%s/%s" % (path,f) for f in l if re.search("\.h5$",f)]
    return files


    
