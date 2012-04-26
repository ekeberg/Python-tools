import required_orientations as ro
import parallel
import rotations
from pylab import *

n = 5

setup_decreased = ro.Setup(83.*(n-1), 83.)
setup = ro.Setup(83.*n, 83.) #415

model_side = setup.get_int_ratio()
number_of_iterations = 50
number_of_repeats = 1000

inserter = ro.SliceInserter(model_side, model_side)
model = zeros((model_side,)*3)

def single_repeat(inserter, number_of_iterations):
    seed()
    model = zeros((model_side,)*3)
    for i in range(number_of_iterations):
        inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
    return model


intermediate_average = []
intermediate_std = []
for batch in range(10):
    print "batch %d" % batch
    jobs = [(inserter, number_of_iterations)]*number_of_repeats
    job_results = parallel.run_parallel(jobs, single_repeat, quiet=True)

    for model in job_results:
        model[model > 0] = 1.

    #summed_model = average(job_results, axis=0)
    intermediate_average.append(average(job_results, axis=0))
    intermediate_std.append(std(job_results, axis=0))


final_average = average(intermediate_average, axis=0)
final_std = std(intermediate_std, axis=0)

final_average_trans = final_average.copy()
final_average_trans[final_average == 0] = nan
final_std_trans = final_std.copy()
final_std_trans[final_average == 0] = nan
