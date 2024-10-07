import time 
import os
import sys
import threading
import math
import ray
import cudf
import numpy as np
from numba import njit
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
    arr = cuda.device_array_like(np.arange(1000, dtype=np.uint16))
    dev_res  = cuda.to_device(np.zeros(1000,dtype=np.uint16))

    tbp = 500
    bpg = len(arr) + (tbp - 1) // tbp

    increment_kernel[bpg, tbp](arr, dev_res)

    print(dev_res.copy_to_host())
@cuda.jit
def increment_kernel(dev_arr, dev_res):
    idx = cuda.threadIdx.x  
    if idx < len(dev_arr):
        dev_res[idx] = dev_arr[idx] + 1
@cuda.jit
def transform_kmers_kernel(dev_local_kmers, dev_res):
    #not sure if this is enough for indexing
    t_idx = cuda.threadIdx.x

    if t_idx < len(dev_local_kmers):
        
        #transform to its corresponding key
        key = 0
        idx = len(dev_local_kmers[t_idx]) - 1
        multiplier = 1
        while idx >= 0:
            key += (dev_local_kmers[t_idx][idx] * multiplier)
            multiplier *= 10
        dev_res[t_idx] = key

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



#this actor has a gpu allocated to it and this actor calls the cudf with its allocated resource in order to prepare the kmer spectrum data
#before operating it using probabilistic model

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

#marks the base index as 1 if erroneous. 0 otherwise
@cuda.jit
def identify_base_trustiness(kmer_spectrum, read , offsets, result, kmer_len):
    threadIdx = cuda.grid(1)
    #find the read assigned to this thread
    start, end = offsets[threadIdx][0], offsets[threadIdx][1] 

    #if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if  threadIdx < offsets.shape[0]:
        
        for idx in range(start + kmer_len - 1, end - kmer_len - 1):
            
            left_portion = read[idx - (kmer_len - 1): idx + 1]
            right_portion = read[idx: idx + kmer_len]

            left_kmer = transform_to_key(left_portion, 5)
            right_kmer = transform_to_key(right_portion, 5)
           
            #refactor kmer_lookup during optimization
            if not (in_spectrum(kmer_spectrum, left_kmer) and in_spectrum(kmer_spectrum, right_kmer)):
                result[idx] = 1

#prone to warp divergence. Too many conditional branches
@cuda.jit
def correct_reads(kmer_spectrum, read, offsets, result, kmer_len):
    threadIdx = cuda.grid(1)
    #find the read assigned to this thread
    start, end = offsets[threadIdx][0], offsets[threadIdx][1] 

    bases = [1,2,3,4]
    #within the range of start and end, find the bases that has value one and then try to correct it 
    
    for idx in range(start, end):
        if result[idx] == 1:
            #base to be corrected
            current_base = read[idx]

            for alternative_base in bases:
                if alternative_base != current_base:

                    left_portion = read[idx - (kmer_len - 1): idx + 1]
                    right_portion = read[idx: idx + kmer_len]
                    left_portion[-1] = alternative_base
                    right_portion[0] = alternative_base

                    candidate_left = transform_to_key(left_portion, kmer_len) 
                    candidate_right = transform_to_key(right_portion, kmer_len) 
                    
                    
                    if (in_spectrum(kmer_spectrum, candidate_left) and in_spectrum(kmer_spectrum, candidate_right)):
                        pass




@ray.remote(num_gpus=1, num_cpus=1)
def remote_two_sided(kmer_spectrum, reads_1d, offsets):

    #transffering necessary data into GPU side
    dev_reads_1d = cuda.to_device(reads_1d)
    dev_kmer_spectrum = cuda.to_device(kmer_spectrum)
    dev_offsets = cuda.to_device(offsets)
    dev_result = cuda.to_device(np.zeros(reads_1d.size, dtype=np.int32))

    #allocating gpu threads
    tbp = 1000
    bpg = (offsets.shape[0] + tbp) // tbp

    #assigns zero to the base if trusted, otherwise 1
    identify_base_trustiness[bpg, tbp](dev_kmer_spectrum, dev_reads_1d, dev_offsets, dev_result, 5)


    return dev_result.copy_to_host()
    
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
def batch_printing(batch_data):
    ones = 0
    for base in batch_data:
        if base == 1:
            ones += 1
    print(ones)
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

        cpus_detected = int(ray.cluster_resources()['CPU'])
        
        gpu_extractor = KmerExtractorGPU.remote(5)
        kmer_occurences = ray.get(gpu_extractor.transform_reads.remote(reads))
        offsets = ray.get(gpu_extractor.get_offsets.remote(reads))
        reads_1d = ray.get(gpu_extractor.transform_reads_2_1d.remote(reads))
        pd_kmers = kmer_occurences.to_pandas()

        occurence_data = kmer_occurences['count'].to_numpy()

        cutoff_threshold = calculatecutoff_threshold(occurence_data, 60)
        filtered_kmer_df = kmer_occurences[kmer_occurences['count'] >= cutoff_threshold]
        kmer_np = filtered_kmer_df['translated'].to_numpy() 
        identified_error_base = ray.get(remote_two_sided.remote(kmer_np, reads_1d, offsets))
        
        # print(identified_error_base)
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

