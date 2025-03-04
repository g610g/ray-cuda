import numpy as np

# size = 3012334
# cpus = 48
# batch_size = size // cpus


def transform_key(km, len):

    multiplier = 1
    key = 0
    while len != 0:
        key += km[len - 1] * multiplier
        multiplier *= 10
        len -= 1

    return key
np_arr = np.array([3,2,2,1,2,4,4,4,4,1,4,3,3,2,3,1,1,3,2] )
# sorted_by_first_col = np_arr[np_arr[:, 0].argsort()]
# sorted_by_second_col = np_arr[np_arr[:, 1].argsort()]
print(transform_key(np_arr, 19))

