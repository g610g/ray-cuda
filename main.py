import numpy as np

# size = 3012334
# cpus = 48
# batch_size = size // cpus

def sort_ping(region_indices, regions_num):
    #already sorted
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

sort_ping(region_indices, regions_num)
print(region_indices)
