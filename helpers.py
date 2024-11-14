from numba import cuda
import ray


#this is a bottleneck for our current implementation.
@cuda.jit(device=True)
def in_spectrum(spectrum, kmer):
    if binary_search_2d(spectrum, kmer) != -1:
        return True

    return False

@cuda.jit(device=True)
def binary_search_2d(sorted_arr, needle):
    sorted_arr_len = len(sorted_arr)
    right = sorted_arr_len - 1
    left = 0

    while(left <= right):
        middle  = (left + right) // 2
        if sorted_arr[middle][0] == needle:
            return middle

        elif sorted_arr[middle][0] > needle:
            right = middle - 1

        elif sorted_arr[middle][0] < needle:
            left = middle + 1
    return -1

@cuda.jit(device=True)
def transform_to_key(ascii_kmer, len):
    multiplier = 1
    key = 0
    while(len != 0):
        key += (ascii_kmer[len - 1] * multiplier)
        multiplier *= 10
        len -= 1

    return key

@cuda.jit(device=True)
def mark_solids_array(solids, start, end):

    for i in range(start, end):
        solids[i] = 1

@cuda.jit(device=True)
def give_kmer_multiplicity(kmer_spectrum, kmer):

    index = binary_search_2d(kmer_spectrum, kmer)
    if index != -1:
        return kmer_spectrum[index][1]

    return -1

@cuda.jit(device=True)
def identify_solid_bases(reads, start, end, kmer_len, kmer_spectrum, solids):

    for idx in range(start, end - (kmer_len - 1)):
        curr_kmer = transform_to_key(reads[idx:idx + kmer_len], kmer_len)

        #set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):

            mark_solids_array(solids, idx - start,  (idx + kmer_len) - start)

#identifying the trusted region in the read and store it into the 2d array
@cuda.jit(device=True)
def identify_trusted_regions(start, end, kmer_spectrum, reads, kmer_len, region_indices, solids):

    identify_solid_bases(reads, start, end, kmer_len, kmer_spectrum, solids)

    current_indices_idx = 0
    base_count = 0
    region_start = 0
    region_end = 0

    for idx in range(end - start):

        #a trusted region has been found
        if base_count >= kmer_len and solids[idx] == -1:

            region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, region_end 
            region_start = idx + 1
            region_end = idx + 1
            current_indices_idx += 1
            base_count = 0

        #reset the region start since its left part is not a trusted region anymore
        if solids[idx] == -1 and base_count < kmer_len:
            region_start = idx + 1
            region_end = idx + 1
            base_count = 0

        if solids[idx] == 1:
            region_end = idx
            base_count += 1

    #ending
    if base_count >= kmer_len:

        region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, region_end
        current_indices_idx += 1

    #this will be the length or the number of trusted regions
    return current_indices_idx

@cuda.jit(device=True)
def copy_solids(threadIdx, solids, arr):

    for idx, base in enumerate(solids):
        arr[threadIdx][idx] = base


#marks the base index as 1 if erroneous. 0 otherwise
@cuda.jit
def benchmark_solid_bases(dev_reads, dev_kmer_spectrum, solids, dev_offsets, kmer_len):
    threadIdx = cuda.grid(1)
    if threadIdx <= dev_offsets.shape[0]:
        start, end = dev_offsets[threadIdx][0], dev_offsets[threadIdx][1]

        for i in range(end - start):
            solids[threadIdx][i] = -1

        identify_solid_bases(dev_reads, start, end, kmer_len, dev_kmer_spectrum, solids[threadIdx])

@ray.remote(num_cpus=1) 
def print_num_not_corrected(not_corrected_counter):
    count = 0
    for nothing in not_corrected_counter:
        if nothing > 0:
            count += nothing
    print(count)
@ray.remote(num_cpus=1) 
def batch_printing(batch_data):

    for idx, error_kmers in enumerate(batch_data):
        for base in error_kmers:
            if base == -1:
                print(error_kmers)
                break

@ray.remote(num_cpus=1) 
def batch_printing_counter(batch_data):

    for idx, corrections in enumerate(batch_data):
        for correction in corrections:
            if correction == 10:
                print("Error but no alternatives")

            if correction == 1:
                print("yehey!!!!")

            if correction > 1:
                print("huhuhu!!!!")

def help(read, kmer_len, offsets):
    for offset in offsets:
        start, end = offset[0], offset[1]
        for idx in range(start + kmer_len - 1, end - (kmer_len - 1)):

            left_portion = read[idx - (kmer_len - 1): idx + 1]
            right_portion = read[idx: idx + kmer_len]
            if len(left_portion) != kmer_len or len(right_portion) != kmer_len:

                print("Naay extracted nga below kmer len")

