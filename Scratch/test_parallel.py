import parallel
import pylab
import timeit

indata = []

def random_image(foo):
    return pylab.random((1000,1000))

stage = """
import parallel
import pylab
import timeit

def random_image(foo):
    start = pylab.random((1000,1000))
    for i in range(5):
        start = pylab.fft2(start)
    return start
"""

program = """
indata = parallel.run_parallel(pylab.zeros(100),random_image,8)
"""


t = timeit.Timer(program,setup=stage).timeit(1)
print t

#timeit.Timer("r = parallel.run_parallel(indata,pylab.fft2,1)").timeit()


    
