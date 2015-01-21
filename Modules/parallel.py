"""Parallelise the execution of a single function with many inputs."""
import multiprocessing
import Queue

class Worker(multiprocessing.Process):
    """Runs a single function for many different outputs."""
    def __init__(self, working_queue, return_dict, process, quiet=False):
        super(Worker, self).__init__()
        self.working_queue = working_queue
        self.return_dict = return_dict
        self.process = process
        self.result = []
        self.quiet = quiet
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
            if not self.quiet:
                try:
                    print "%s process %d, approx %d left" % (self.name, tmp[0], self.working_queue.qsize())
                except NotImplementedError:
                    print "%s process %d" % (self.name, tmp[0])
            print tmp[0]
            print self.process
            print tmp[1]
            self.return_dict[tmp[0]] = self.process(*tmp[1])
        if not self.quiet:
            print "%s done" % self.name


def run_parallel(jobs, function, n_cpu=0, quiet=False):
    """Execute the function for each input given in the array jobs and return the results in an array.
    Jobs must be iterable and the jobs should be a tuple containing the function arguments."""
    if not n_cpu:
        n_cpu = multiprocessing.cpu_count()
    working_queue = multiprocessing.Queue()
    #return_queue = multiprocessing.Queue()
    my_manager = multiprocessing.Manager()
    return_dict = my_manager.dict()
    workers = []
    for i, job in enumerate(jobs):
        if isinstance(job, tuple):
            working_queue.put((i, job))
        else:
            working_queue.put((i, (job, )))

    for i in range(n_cpu):
        workers.append(Worker(working_queue, return_dict, function, quiet))
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    values = return_dict.values()
    my_manager.shutdown()
    return values

def function_square(value):
    """Return the square of the input."""
    return value**2
