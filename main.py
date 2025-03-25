from os import readlink
import numpy as np

# size = 3012334
# cpus = 48
# batch_size = size // cpus


def sort_ping(region_indices, regions_num):
    # already sorted
    if regions_num == 1:
        return

    for i in range(1, regions_num):
        key = region_indices[i]
        j = i - 1
        while j >= 0 and key[2] > region_indices[j][2]:
            region_indices[j + 1] = region_indices[j]
            j -= 1
        region_indices[j + 1] = key


region_indices = [[1, 3, 2], [5, 10, 5], [10, 25, 5], [30, 50, 20]]
regions_num = 4
read_length = 222331233312332
gpu_num = 3
base_size = read_length // gpu_num
remainder = read_length % gpu_num
print(remainder)
start = 0
start_end = []
for i in range(gpu_num):
    extra = 1 if i < remainder else 0  # Distribute remainder to first GPUs
    end = start + base_size + extra
    start_end.append([start, end, end - start])
    start = end

print(start_end)
sort_ping(region_indices, regions_num)
print(region_indices)
