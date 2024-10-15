import time 
import os
import sys
import math
from numpy.core.multiarray import dtype
import ray
import cudf
import numpy as np
from numba import cuda
from Bio import SeqIO
# import seaborn as sns
# import matplotlib.pyplot as plt
import pandas as pd

ray.init()

@ray.remote(num_cpus=1)
def count_occurence(kmers, occurence_dictionary):
    for key in kmers:
        if key in occurence_dictionary:
            occurence_dictionary[key] += 1
        else:
            occurence_dictionary[key] = 1
@ray.remote(num_gpus=1)
class GPUActor:
    def ping(self):
        print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

#this task will be run into node that has GPU accelerator
@ray.remote(num_gpus=1)
def gpu_task():
    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))



@ray.remote(num_gpus=1)
def run_jitted_cuda():
    arr = cuda.device_array_like(np.arange(10000, dtype=np.uint16))

    tpb = 1000
    bpg = len(arr) + (tpb- 1) // tpb

    increment_kernel[bpg, tpb](arr)

    return (arr.copy_to_host())

@cuda.jit(device=True)
def increment_device(idx, global_arr):
    global_arr[idx] += 2

@cuda.jit
def increment_kernel(dev_arr):
    idx = cuda.grid(1)
    if idx < len(dev_arr):
       increment_device(idx, dev_arr) 


# simple transpose communication pattern
@cuda.jit
def transpose_comm(dev_arr, output_arr):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
     
    output_arr[ty][tx] = dev_arr[tx][ty]

@cuda.jit
def sync(dev_arr, res_arr, sum):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    res_arr[ty][tx] = dev_arr[tx][ty]

    #a barrier that will only proceed code when all of the threads running this kernel has done executing
    cuda.synchronize()

    sum[tx + ty] = res_arr[tx][ty]


#the linear search for kmer
@cuda.jit(device=True)
def in_spectrum(spectrum, kmer):
    for k in spectrum:
        if kmer == k:
            return True
    return False

@cuda.jit(device=True)
def transform_to_key(ascii_kmer, len):
    multiplier = 1
    key = 0
    while(len != 0):
        key += (ascii_kmer[len - 1] * multiplier)
        multiplier *= 10
        len -= 1

    return key
#start is the starting kmer point
#absolute_start is the read absolute starting point from the 1d read
@cuda.jit(device=True)
def mark_solids_array(solids, start, end):

    for i in range(start, end):
        solids[i] = 1

#for correcting the edge bases in two-sided correction (not sure of the implementation yet)
#only corrects the left edges of the read
@cuda.jit(device=True)
def correct_edge_bases(idx, read, kmer, bases, kmer_len, kmer_spectrum, threadIdx, counter, corrected_counter):
    current_base = read[idx]
    posibility = 0
    candidate = -1
    for alternative_base in bases:
        if alternative_base != current_base:

            kmer[0] = alternative_base
            candidate_kmer = transform_to_key(kmer, kmer_len)

            if in_spectrum(kmer_spectrum, candidate_kmer):
                posibility += 1
                candidate = alternative_base

    if posibility == 1:
        read[idx] = candidate
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    if posibility == 0:
        corrected_counter[threadIdx][counter] = 10
        counter += 1

    if  posibility > 1:
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    return counter
@cuda.jit(device=True)
def correct_reads(idx, read, kmer_len,kmer_spectrum, corrected_counter, bases, left_kmer, right_kmer, threadIdx, counter):
    current_base = read[idx]
    posibility = 0
    candidate = -1

    for alternative_base in bases:
        if alternative_base != current_base:

            #array representation
            left_kmer[-1] = alternative_base
            right_kmer[0] = alternative_base

            #whole number representation
            candidate_left = transform_to_key(left_kmer, kmer_len)
            candidate_right = transform_to_key(right_kmer, kmer_len)

            #the alternative base makes our kmers trusted
            if in_spectrum(kmer_spectrum, candidate_left) and in_spectrum(kmer_spectrum, candidate_right):
                posibility += 1
                candidate = alternative_base

    if posibility == 1:
        read[idx] = candidate
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    if posibility == 0:
        corrected_counter[threadIdx][counter] = 10
        counter += 1

    if  posibility > 1:
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    return counter
#marks the base index as 1 if erroneous. 0 otherwise

@cuda.jit
def two_sided_kernel(kmer_spectrum, read, offsets, result, kmer_len, corrected_counter):
    threadIdx = cuda.grid(1)
    counter = 0
    
    #if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if threadIdx < offsets.shape[0]:

        #find the read assigned to this thread
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        MAX_LEN = 256 
        bases = cuda.local.array(5, dtype='uint8')
        solids = cuda.local.array(MAX_LEN, dtype='int8')

        for i in range(end - start):
            solids[i] = -1

        for i in range(5):
            bases[i] = i + 1

        #identify whether base is solid or not
        for idx in range(start, end - (kmer_len - 1)):
            curr_kmer = transform_to_key(read[idx:idx + kmer_len], kmer_len)

            #set the bases as solids
            if in_spectrum(kmer_spectrum, curr_kmer):

                # mark_solids_array(solids, idx, idx + kmer_len, start)
                mark_solids_array(solids, idx - start,  (idx + kmer_len) - start)

        #used for debugging
        for idx in range(end - start):
            result[threadIdx][idx] = solids[idx]

        #check whether base is potential for correction
        #kulang pani diria sa pag check sa first and last bases
        for base_idx in range(end - start):
            #the base needs to be corrected
            if solids[base_idx] == -1 and base_idx >= (kmer_len - 1) and base_idx <= (end - start) - kmer_len:

                left_portion = read[(base_idx + start) - (kmer_len - 1): base_idx + start + 1]
                right_portion = read[base_idx + start: base_idx + kmer_len]

                counter = correct_reads(base_idx + start, read, kmer_len, kmer_spectrum, corrected_counter, bases, left_portion, right_portion, threadIdx, counter)

            #the leftmost bases of the read
            if solids[base_idx] == -1 and base_idx < (kmer_len - 1):
                pass
            #the rightmost bases of the read
            if solids[base_idx] == -1 and base_idx > (end - start) - kmer_len:
                pass

@ray.remote(num_gpus=1, num_cpus=1)
def remote_two_sided(kmer_spectrum, reads_1d, offsets, kmer_len):
    cuda.profile_start()
    start = cuda.event()
    end = cuda.event()

    start.record()
    #transffering necessary data into GPU side
    dev_reads_1d = cuda.to_device(reads_1d)
    dev_kmer_spectrum = cuda.to_device(kmer_spectrum)
    dev_offsets = cuda.to_device(offsets)
    dev_result = cuda.to_device(np.zeros((offsets.shape[0], 256), dtype='uint64'))
    dev_corrected_counter = cuda.to_device(np.zeros((len(offsets), 50), dtype='uint64'))

    #allocating gpu threads
    tbp = 1024
    bpg = (offsets.shape[0] + tbp) // tbp

    # #assigns zero to the base if trusted, otherwise 1
    two_sided_kernel[bpg, tbp](dev_kmer_spectrum, dev_reads_1d, dev_offsets , dev_result, kmer_len, dev_corrected_counter)

    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the kernel:  {transfer_time} ms")

    h_corrected_counter = dev_corrected_counter.copy_to_host()
    cuda.profile_stop()
    return [h_corrected_counter ,dev_result.copy_to_host()]

@cuda.jit(device=True)
def correct_read_one_sided_right(reads, start, region_end, kmer_spectrum, kmer_len, bases, alternatives):

    alternative_counter = 0
    possibility = 0
    alternative = -1
    curr_kmer = reads[start + (region_end - (kmer_len - 1)): region_end + start + 1]
    forward_kmer = reads[start + (region_end - (kmer_len - 1)) + 1: region_end + start + 2]

    curr_kmer_transformed = transform_to_key(curr_kmer, kmer_len) 
    forward_kmer_transformed = transform_to_key(forward_kmer, kmer_len) 

    #we can now correct this. else return diz shet or break 
    if in_spectrum(kmer_spectrum, curr_kmer_transformed) and not in_spectrum(kmer_spectrum, forward_kmer_transformed):
        #find alternative  base

        for alternative_base in bases:
            forward_kmer[-1] = alternative_base
            candidate_kmer = transform_to_key(forward_kmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):

                alternatives[possibility][0], alternatives[possibility][1] = alternative_base, 4000
                possibility += 1
                alternative = alternative_base

    #not sure if correct indexing for reads
    if possibility == 1:
        reads[region_end + start + 1] = alternative
        return True

    #Break the correction since it fails to correct the read
    if possibility == 0:
        return False

    #we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        max = 0
        for idx in range(alternative_counter):
            if alternatives[idx][1] + 1 > alternatives[max][1] + 1:
                max = idx

        reads[region_end + start + 1] = alternatives[max][0]
        return True

#identifying the trusted region in the read and store it into the 2d array
@cuda.jit(device=True)
def identify_trusted_regions(start, end, kmer_spectrum, reads, kmer_len, region_indices, solids):

    #we can use the solids array from the two sided (not yet done)
    for idx in range(start, end - (kmer_len - 1)):
            curr_kmer = transform_to_key(reads[idx:idx + kmer_len], kmer_len)

            #set the bases as solids
            if in_spectrum(kmer_spectrum, curr_kmer):

                # mark_solids_array(solids, idx, idx + kmer_len, start)
                mark_solids_array(solids, idx - start,  (idx + kmer_len) - start)

    current_indices_idx = 0
    base_count = 0
    region_start = 0
    prefix = 0

    for idx in range(end - start):

        #a trusted region has been found
        if base_count >= kmer_len and solids[idx] == -1:

            region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, prefix
            region_start = idx + 1
            current_indices_idx += 1
            prefix = idx + 1
            base_count = 0

        #reset the region start since its left part is not a trusted region anymore
        if solids[idx] == -1 and base_count < kmer_len:
            region_start = idx + 1
            prefix = idx + 1
            base_count = 0

        if solids[idx] == 1:
            prefix = idx
            base_count += 1

    #ending
    if base_count >= kmer_len:

        region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, prefix
        current_indices_idx += 1

    #this will be the length or the number of trusted regions
    return current_indices_idx

@cuda.jit
def one_sided_kernel(kmer_spectrum, reads_1d, offsets, kmer_len):
    threadIdx = cuda.grid(1) 
    
    if threadIdx < offsets.shape[0]:

        MAX_LEN = 256
        region_indices = cuda.local.array((10,2), dtype="int8")
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        solids = cuda.local.array(MAX_LEN, dtype='int8')
        alternatives = cuda.local.array((4, 2) dtype='uint32')

        for i in range(end - start):
            solids[i] = -1

        bases = cuda.local.array(5, dtype='uint8')
        regions_count = identify_trusted_regions(start, end, kmer_spectrum, reads_1d, kmer_len, region_indices, solids)

        #fails to correct the read (how about regions that has no error?)
        if regions_count == 0:
            return

        for region in range(regions_count):
            #going to the right first

            #there is no next region
            if region + 1 == regions_count:

                region_start, region_end = region_indices[region][0], region_indices[region][1]
                while region_end + 1 != (end - start):
                    curr_kmer = reads_1d[start + (region_end - (kmer_len - 1)): region_end + start + 1]

            #there is a next region
            if region + 1 != regions_count:
                region_start, region_end = region_indices[region][0], region_indices[region][1]
                next_region_start, next_region_end = region_indices[region + 1][0], region_indices[region + 1][1]
                while region_end != (next_region_start - 1):
                    pass


@ray.remote(num_gpus=1, num_cpus=1)
class KmerExtractorGPU:
    def __init__(self, kmer_length):
        self.kmer_length = kmer_length
        self.translation_table = str.maketrans({"A" : "1", "C":"2", "G":"3", "T":"4", "N":"5"})
    def create_kmer_df(self, reads):
        read_df = cudf.Series(reads)
        kmers = read_df.str.character_ngrams(self.kmer_length, True)
        exploded_kmers = kmers.explode()
        return exploded_kmers.value_counts()

    def get_offsets(self, reads):
        read_df = cudf.DataFrame({'reads':reads})
        str_lens = read_df['reads'].str.len()
        end_indices = str_lens.cumsum()
        start_indices = end_indices.shift(1, fill_value=0)
        offsets = cudf.DataFrame({'start_indices':start_indices, 'end_indices':end_indices}).to_numpy()
        return offsets

    def transform_reads_2_1d(self, reads):
        read_df = cudf.DataFrame({'reads':reads})
        return read_df['reads'].str.findall('.').explode().str.translate(self.translation_table).astype('int16').to_numpy()

    def get_read_lens(self, reads):
        read_df = cudf.DataFrame({'reads':reads})
        return read_df['reads'].str.len()
    def transform_reads(self, reads):
        read_s = cudf.Series(reads, name='reads')
        read_df = read_s.to_frame()

        read_df['translated'] = read_df['reads'].str.translate(self.translation_table)
        ngram_kmers = read_df['translated'].str.character_ngrams(self.kmer_length, True)

        exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
        numeric_ngrams  = exploded_ngrams.astype('int32').reset_index(drop=True)
        result_frame = numeric_ngrams.value_counts().reset_index()

        result_frame.columns = ['translated', 'count']

        return result_frame

#problem:what if the lowest is at the left side of the value or in the spurious area?
def calculatecutoff_threshold(occurence_data, bin):

    hist_vals, bin_edges = np.histogram(occurence_data, bins=bin, density=False)

    bin_centers = (0.5 * (bin_edges[1:] + bin_edges[:-1]))

    valley_index = 1
     
    for idx in range(valley_index + 1, len(hist_vals)):
        if hist_vals[idx] < hist_vals[valley_index]:
            valley_index = idx
            continue
        break
  
    peak_index = valley_index + 1 
    for idx in range(peak_index + 1, len(hist_vals)):
        if hist_vals[idx] > hist_vals[peak_index]:
            peak_index = idx
            continue
        break
  
    min_density_idx = valley_index 
    for idx in range(valley_index, peak_index + 1):
        if hist_vals[idx] < hist_vals[min_density_idx]:
            min_density_idx = idx
    return math.ceil(bin_centers[min_density_idx])

@ray.remote(num_cpus=1) 
def batch_printing(batch_data, kmer_spectrum):

    for idx, error_kmers in enumerate(batch_data):
        print(error_kmers)

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
@ray.remote(num_gpus=1) 
class GPUPing:
    def ping(self):
        print(ray.get_runtime_context().get_accelerator_ids()) 
@ray.remote(num_gpus=1) 
def ping_resources():
   print(ray.cluster_resources()) 
if __name__ == '__main__':
        start_time = time.perf_counter()
        usage = "Usage " + sys.argv[0] + " <FASTQ file>"
        if len(sys.argv) != 2:
            print(usage)
            exit(1)

        with open(sys.argv[1]) as handle:
            fastq_data = SeqIO.parse(handle, 'fastq')
            reads = [str(data.seq) for data in fastq_data]

        kmer_len = 5
        cpus_detected = int(ray.cluster_resources()['CPU'])
        gpu_extractor = KmerExtractorGPU.remote(kmer_len)
        kmer_occurences = ray.get(gpu_extractor.transform_reads.remote(reads))
        offsets = ray.get(gpu_extractor.get_offsets.remote(reads))
        reads_1d = ray.get(gpu_extractor.transform_reads_2_1d.remote(reads))
        pd_kmers = kmer_occurences.to_pandas()
        
        occurence_data = kmer_occurences['count'].to_numpy()
        
        cutoff_threshold = calculatecutoff_threshold(occurence_data, 60)
        # print(f"cutoff threshold: {cutoff_threshold}")
        

        filtered_kmer_df = kmer_occurences[kmer_occurences['count'] >= cutoff_threshold]
        kmer_np = filtered_kmer_df['translated'].to_numpy() 
        # help(reads_1d, kmer_len, offsets)
        [corrected_counter, result_counter] = ray.get(remote_two_sided.remote(kmer_np, reads_1d, offsets, kmer_len))

        # print(f"marked number of alternatives for a base: {identified_error_base}") 
        batch_size = len(corrected_counter) // cpus_detected

        # print("Result counter :")
        # ray.get([batch_printing.remote((result_counter[batch_len: batch_len + batch_size]), kmer_np) for batch_len in range(0, len(result_counter), batch_size)])

        print("Corrected counter :")
        ray.get([batch_printing_counter.remote((corrected_counter[batch_len: batch_len + batch_size])) for batch_len in range(0, len(corrected_counter), batch_size)])
        #visuals
        # plt.figure(figsize=(12, 6))
        # sns.histplot(pd_kmers['count'], bins=60, kde=True, color='blue', alpha=0.7)
        # plt.title('K-mer Coverage Histogram')
        # plt.xlabel('K-mer Occurrence (Multiplicity)')
        # plt.ylabel('Density (Number of k-mers with same multiplicity)')
        # plt.grid(True)
        # plt.show()

        end_time = time.perf_counter()
        print(f"Elapsed time is {end_time - start_time}")

