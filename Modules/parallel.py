
import sys
import multiprocessing
import Queue
from time import sleep
#from guppy import hpy

class Worker(multiprocessing.Process):
    def __init__(self, working_queue, return_dict, process):
        multiprocessing.Process.__init__(self)
        self.working_queue = working_queue
        self.return_dict = return_dict
        self.process = process
        self.result = []
    def run(self):
        # while not self.working_queue.empty():
        #     try:
        #         print "%s get new, approx %d left" % (self.name,self.working_queue.qsize())
        #     except NotImplementedError:
        #         pass
        #     try:
        #         i,data = self.working_queue.get_nowait()
        #         print self.name, " got data ", i
        #     except Queue.Empty:
        #         print self.name, " didn't get data"
        #         break
        #     self.return_dict[i] = self.process(*data)
        # print "%s done" % self.name
        # return
        while True:
            try:
                tmp = self.working_queue.get(timeout=0.1) #timeout of 0.1 is choosen adhoc.
            except Queue.Empty:
                break
            if tmp == None:
                break
            try:
                print "%s process %d, approx %d left" % (self.name, tmp[0], self.working_queue.qsize())
            except NotImplementedError:
                print "%s process %d" % (self.name, tmp[0])
                pass
            self.return_dict[tmp[0]] = self.process(*tmp[1])
        print "%s done" % self.name


def run_parallel(jobs, function, n_cpu=0):
    """Execute the function for each input given in the array jobs and return the results in an array."""
    if not n_cpu: n_cpu = multiprocessing.cpu_count()
    working_queue = multiprocessing.Queue()
    #return_queue = multiprocessing.Queue()
    my_manager = multiprocessing.Manager()
    return_dict = my_manager.dict()
    workers = []
    for job,i in zip(jobs,range(len(jobs))):
        working_queue.put((i,job))

    # for i in range(n_cpu):
    #     #Worker(working_queue, return_queue, function).start()
    #     workers.append(Worker(working_queue, return_dict, function))
    #     workers[-1].start()
    #     #print workers[-1].is_alive()
    for i in range(n_cpu):
        workers.append(Worker(working_queue, return_dict, function))
    for w in workers:
        w.start()
        
    for w in workers:
        w.join()

    #     w.terminate()
    #     print w.name, " ", w.is_alive()
    # results = []
    # while not return_queue.empty():
    #     results.append(return_queue.get())
    # return_queue.close()
    # working_queue.close()

    # results.sort()
    # return [r[1] for r in results]
    # return results
    values = return_dict.values()
    my_manager.shutdown()
    return values

def f(x):
    return x**2
