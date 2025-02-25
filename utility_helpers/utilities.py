import ray
from numba import cuda


@ray.remote(num_cpus=1)
def calculate_solids(solids):
    for solid in solids:
        solids_count = 0
        for base in solid:
            if base == -1:
                solids_count += 1

        if solids_count > 0:
            print(solids_count)


@ray.remote(num_cpus=1)
def check_votes(votes):
    has_votes = 0
    for vote in votes:
        for item in vote:
            if item > 0:
                has_votes += 1
                break

    print(has_votes)


@cuda.jit(device=True)
def test_return_value(val, iter, bases):
    sum = 0
    for base in bases:
        sum += base
    return sum

@cuda.jit()
def reverse_comp(reverse_comp, kmers):
    threadIdx = cuda.grid(1)
    if threadIdx < kmers.shape:
        


