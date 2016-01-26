from mpi4py import MPI as _MPI

WORKTAG = 0
DIETAG = 1

def slave(function):
    comm = _MPI.COMM_WORLD
    status = _MPI.Status()
    while 1:
        data = comm.recv(obj=None, source=0, tag=_MPI.ANY_TAG, status=status)
        if status.Get_tag():
            break
        result = function(data[1])
        comm.send(obj=(data[0], result), dest=0)


def master(jobs):
    comm = _MPI.COMM_WORLD
    size = comm.Get_size()
    if len(jobs) < size:
        active_size = len(jobs)
    else:
        active_size = size
    status = _MPI.Status()
    job_stack = [(i, v) for i, v in enumerate(jobs)]
    all_data = []

    for i in range(1, active_size):
        try:
            next_job = job_stack.pop()
        except IndexError:
            break
        comm.send(obj=next_job, dest=i, tag=WORKTAG)

    for i in range(active_size, size):
        comm.send(obj=None, dest=i, tag=DIETAG)

    while 1:
        try:
            next_job = job_stack.pop()
        except IndexError:
            break
        data = comm.recv(obj=None, source=_MPI.ANY_SOURCE, tag=_MPI.ANY_TAG, status=status)
        all_data.append(data)
        comm.send(obj=next_job, dest=status.Get_source(), tag=WORKTAG)

    for i in range(1, active_size):
        data = comm.recv(obj=None, source=_MPI.ANY_SOURCE, tag=_MPI.ANY_TAG)
        all_data.append(data)

    for i in range(1, active_size):
        comm.send(obj=None, dest=i, tag=DIETAG)
        
    sorted_data = sorted(all_data, key=lambda x: x[0])
    data_no_index = [data[1] for data in sorted_data]
    return data_no_index

def run_in_parallel(function, jobs):
    comm = _MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        result = master(jobs)
        return result
    else:
        slave(function)
    

def is_master():
    comm = _MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        return True
    else:
        return False

def number_of_slaves():
    comm = _MPI.COMM_WORLD
    size = comm.Get_size()
    return size - 1

def rank():
    comm = _MPI.COMM_WORLD
    rank = comm.Get_rank()
    return rank
    
