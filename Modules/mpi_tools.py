from mpi4py import MPI
import collections

WORKTAG = 0
DIETAG = 1

def slave(function):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    while 1:
        data = comm.recv(obj=None, source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag():
            break
        result = function(data[1])
        comm.send(obj=(data[0], result), dest=0)


def master(jobs):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    status = MPI.Status()
    job_stack = [(i, v) for i, v in enumerate(jobs)]
    all_data = []

    for i in range(1, size):
        try:
            next_job = job_stack.pop()
        except IndexError:
            break
        comm.send(obj=next_job, dest=i, tag=WORKTAG)

    while 1:
        try:
            next_job = job_stack.pop()
        except IndexError:
            break
        data = comm.recv(obj=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        all_data.append(data)
        comm.send(obj=next_job, dest=status.Get_source(), tag=WORKTAG)

    for i in range(1, size):
        data = comm.recv(obj=None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        all_data.append(data)

    for i in range(1, size):
        comm.send(obj=None, dest=i, tag=DIETAG)

        
    sorted_data = sorted(all_data, key=lambda x: x[0])
    data_no_index = [data[1] for data in sorted_data]
    return data_no_index

def run_in_parallel(function, jobs):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        result = master(jobs)
        return result
    else:
        slave(function)
    
