import time 
import os
import sys
import math
from numba.cuda.device_init import threadIdx
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
        if kmer == k[0]:
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
def give_kmer_multiplicity(kmer_spectrum, kmer):
    for k in kmer_spectrum:
        if k[0] == kmer:
            return k[1]
    return -1
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

@cuda.jit(device=True)
def invoke_voting(vm, kmer_spectrum, bases, kmer_len, curr_idx, read, start):

    curr_kmer = read[curr_idx: curr_idx + kmer_len]
    for idx in range(kmer_len):
        for base in bases:
            curr_kmer[idx] = base
            trans_curr_kmer = transform_to_key(curr_kmer, kmer_len)

            #add a vote to the corresponding index
            if in_spectrum(kmer_spectrum, trans_curr_kmer):
                vm[base - 1][(curr_idx - start) + idx] += 1

    #i cant see the produced voting matrix
#the voting refinement
@cuda.jit
def voting_algo(dev_reads, offsets, kmer_spectrum, kmer_len):
    threadIdx = cuda.grid(1)

    if threadIdx <= offsets.shape[0]:

        MAX_LEN = 256 
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        vm = cuda.local.array((4, MAX_LEN), 'uint16')
        bases = cuda.local.array(4, 'uint8')

        for idx in range(4):
            bases[idx] = idx + 1

        for idx in range(start, end - (kmer_len - 1)):
            curr_kmer = transform_to_key(reads[idx:idx + kmer_len], kmer_len)

            #invoke voting if the kmer is not in spectrum
            if not in_spectrum(kmer_spectrum, curr_kmer):
                invoke_voting(vm, kmer_spectrum, bases, kmer_len, idx, dev_reads, start)


#marks the base index as 1 if erroneous. 0 otherwise
@cuda.jit
def benchmark_solid_bases(dev_reads, dev_kmer_spectrum, solids, dev_offsets, kmer_len):
    threadIdx = cuda.grid(1)
    if threadIdx <= dev_offsets.shape[0]:
        start, end = dev_offsets[threadIdx][0], dev_offsets[threadIdx][1]

        for i in range(end - start):
            solids[threadIdx][i] = -1

        identify_solid_bases(dev_reads, start, end, kmer_len, dev_kmer_spectrum, solids[threadIdx])

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
        identify_solid_bases(read, start, end, kmer_len, kmer_spectrum, solids)

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
def remote_two_sided(kmer_spectrum, reads_1d, offsets, kmer_len, two_sided_iter, one_sided_iter, num_cpus):
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
    dev_solids = cuda.to_device(np.zeros((offsets.shape[0], 256), dtype='int8'))
    dev_solids_after = cuda.to_device(np.zeros((offsets.shape[0], 256), dtype='int8'))
    not_corrected_counter = cuda.to_device(np.zeros(offsets.shape[0], dtype='int16'))
    #allocating gpu threads
    tbp = 1024
    bpg = (offsets.shape[0] + tbp) // tbp

    #benchmarking only
    print("Before two sided")
    benchmark_solid_bases[bpg, tbp](dev_reads_1d, dev_kmer_spectrum, dev_solids, dev_offsets, kmer_len)
    h_solids = dev_solids.copy_to_host()
    batch_size = (len(h_solids) // num_cpus) - 4
    print(f"{sum(ray.get([count_error_reads.remote((h_solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(h_solids), batch_size)]))} base errors found") 

    for _ in range(two_sided_iter):

        two_sided_kernel[bpg, tbp](dev_kmer_spectrum, dev_reads_1d, dev_offsets , dev_result, kmer_len, dev_corrected_counter)

    print("After two sided")
    benchmark_solid_bases[bpg, tbp](dev_reads_1d, dev_kmer_spectrum, dev_solids, dev_offsets, kmer_len)
    h_solids = dev_solids.copy_to_host()
    print(f"{sum(ray.get([count_error_reads.remote((h_solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(h_solids), batch_size)]))} base errors found") 

    for _ in range(one_sided_iter):
        one_sided_kernel[bpg, tbp](dev_kmer_spectrum, dev_reads_1d, dev_offsets, kmer_len, dev_solids, dev_solids_after, not_corrected_counter)

    benchmark_solid_bases[bpg, tbp](dev_reads_1d, dev_kmer_spectrum, dev_solids, dev_offsets, kmer_len)
    h_solids = dev_solids.copy_to_host()
    print("After one sided")
    print(f"{sum(ray.get([count_error_reads.remote((h_solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(h_solids), batch_size)]))} base errors found") 
     
    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the kernel:  {transfer_time} ms")

    cuda.profile_stop()

@cuda.jit(device=True)
def correct_read_one_sided_left(reads, start, region_start, kmer_spectrum, kmer_len, bases, alternatives):
    possibility = 0
    alternative = -1

    curr_kmer = reads[start + region_start:(start + region_start) + kmer_len]
    backward_kmer = reads[(start + region_start ) - 1: (start + region_start) + (kmer_len - 1)]

    curr_kmer_transformed = transform_to_key(curr_kmer, kmer_len) 
    backward_kmer_transformed = transform_to_key(backward_kmer, kmer_len) 

    if in_spectrum(kmer_spectrum, curr_kmer_transformed) and not in_spectrum(kmer_spectrum, backward_kmer_transformed):
        #find alternative  base
        for alternative_base in bases:
            backward_kmer[0] = alternative_base
            candidate_kmer = transform_to_key(backward_kmer, kmer_len)

            if in_spectrum(kmer_spectrum, candidate_kmer):

                #alternative base and its corresponding kmer count
                alternatives[possibility][0], alternatives[possibility][1] = alternative_base, give_kmer_multiplicity(kmer_spectrum, candidate_kmer)
                possibility += 1
                alternative = alternative_base

    #Break the correction since it fails to correct the read
    if possibility == 0:
        return False

    #not sure if correct indexing for reads
    if possibility == 1:
        reads[(start + region_start) - 1] = alternative
        return True


    #we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        max = 0
        for idx in range(possibility):
            if alternatives[idx][1] + 1 >= alternatives[max][1] + 1:
                max = idx

        reads[(start + region_start) - 1] = alternatives[max][0]
        return True

@cuda.jit(device=True)
def correct_read_one_sided_right(reads, start, region_end, kmer_spectrum, kmer_len, bases, alternatives):

    possibility = 0
    alternative = -1

    curr_kmer = reads[start + (region_end - (kmer_len - 1)): region_end + start + 1]
    forward_kmer = reads[start + (region_end - (kmer_len - 1)) + 1: region_end + start + 2]

    curr_kmer_transformed = transform_to_key(curr_kmer, kmer_len) 
    forward_kmer_transformed = transform_to_key(forward_kmer, kmer_len) 

    #we can now correct this. else return diz shet or break 
    #if false does it imply failure?
    if in_spectrum(kmer_spectrum, curr_kmer_transformed) and not in_spectrum(kmer_spectrum, forward_kmer_transformed):

        #find alternative  base
        for alternative_base in bases:
            forward_kmer[-1] = alternative_base
            candidate_kmer = transform_to_key(forward_kmer, kmer_len)

            if in_spectrum(kmer_spectrum, candidate_kmer):

                #alternative base and its corresponding kmer count
                alternatives[possibility][0], alternatives[possibility][1] = alternative_base, give_kmer_multiplicity(kmer_spectrum, candidate_kmer)
                possibility += 1
                alternative = alternative_base

    #break the correction since it fails to correct the read
    if possibility == 0:
        return False

    #not sure if correct indexing for reads
    if possibility == 1:
        reads[region_end + start + 1] = alternative
        return True

    

    #we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        max = 0
        for idx in range(possibility):
            if alternatives[idx][1] + 1 >= alternatives[max][1] + 1:
                max = idx

        reads[region_end + start + 1] = alternatives[max][0]
        return True

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

#no implementation for tracking how many corrections are done for each kmers in the read
@cuda.jit
def one_sided_kernel(kmer_spectrum, reads_1d, offsets, kmer_len, solids_counter, solids_after, not_corrected_counter):
    threadIdx = cuda.grid(1) 

    if threadIdx < offsets.shape[0]:

        MAX_LEN = 256
        region_indices = cuda.local.array((10,2), dtype="int8")
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        solids = cuda.local.array(MAX_LEN, dtype='int8')
        alternatives = cuda.local.array((4, 2), dtype='uint32')
        corrected_solids = cuda.local.array(MAX_LEN, dtype='int8')

        for i in range(end - start):
            solids[i] = -1

        for i in range(end - start):
            corrected_solids[i] = -1

        bases = cuda.local.array(5, dtype='uint8')

        for i in range(5):
            bases[i] = i + 1

        regions_count = identify_trusted_regions(start, end, kmer_spectrum, reads_1d, kmer_len, region_indices, solids)

        copy_solids(threadIdx, solids, solids_counter)
        #fails to correct the read does not have a trusted region (how about regions that has no error?)
        if regions_count == 0:
            return

        for region in range(regions_count):
            #going towards right of the region 

            #there is no next region
            if region  == (regions_count - 1):
                region_end = region_indices[region][1]
                while region_end  != ((end - start) - 1):
                    if not correct_read_one_sided_right(reads_1d, start, region_end, kmer_spectrum, kmer_len, bases, alternatives):
                        not_corrected_counter[threadIdx] += 1
                        break
                    else:
                        region_end += 1
                        region_indices[region][1] = region_end

            #there is a next region
            if region  != (regions_count - 1):
                region_end = region_indices[region][1]
                next_region_start = region_indices[region + 1][0] 

                while region_end != (next_region_start - 1):
                    if not correct_read_one_sided_right(reads_1d, start, region_end, kmer_spectrum, kmer_len, bases, alternatives):
                        not_corrected_counter[threadIdx] += 1
                        break
                    else:
                        region_end += 1
                        region_indices[region][1] = region_end

            #going towards left of the region

            #we are the leftmost region
            if region - 1 == -1:
                region_start = region_indices[region][0]

                #while we are not at the first base of the read
                while region_start != 0:
                    if not correct_read_one_sided_left(reads_1d, start, region_start, kmer_spectrum, kmer_len, bases, alternatives):
                        not_corrected_counter[threadIdx] += 1
                        break
                    else:
                        region_start -= 1
                        region_indices[region][0] = region_start

            #there is another region in the left side of this region 
            if region - 1 != -1:
                region_start, prev_region_end = region_indices[region][0], region_indices[region - 1][1]
                while region_start - 1 != (prev_region_end):

                    if not correct_read_one_sided_left(reads_1d, start, region_start, kmer_spectrum, kmer_len, bases, alternatives):
                        not_corrected_counter[threadIdx] += 1
                        break
                    else:
                        region_start -= 1
                        region_indices[region][0] = region_start

        identify_solid_bases(reads_1d, start, end, kmer_len, kmer_spectrum, corrected_solids)
        copy_solids(threadIdx, corrected_solids, solids_after)
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

#returns then number of error bases
@ray.remote(num_cpus=1) 
def count_error_reads(reads):
    count = 0
    for read in reads:
        for base in read:
            if base == -1:
                count += 1
                continue
    return count
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

        kmer_len = 6
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
        kmer_np = filtered_kmer_df.to_numpy() 
        # help(reads_1d, kmer_len, offsets)
        ray.get(remote_two_sided.remote(kmer_np, reads_1d, offsets, kmer_len, 2, 4, cpus_detected))
        # print(f"marked number of alternatives for a base: {identified_error_base}") 
        
        # print("Result counter :")
        # ray.get([batch_printing.remote((result_counter[batch_len: batch_len + batch_size]), kmer_np) for batch_len in range(0, len(result_counter), batch_size)])
        # ray.get([print_num_not_corrected.remote((not_corrected_counter[batch_len: batch_len + batch_size])) for batch_len in range(0, len(not_corrected_counter), batch_size)])
        # print("Before")
        #print(sum(ray.get([count_error_reads.remote((before_solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(before_solids), batch_size)]))) 
        #
        # print("After")
        # print(sum(ray.get([count_error_reads.remote((after_solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(after_solids), batch_size)])) ) 

        # print("Corrected counter :")
        # ray.get([batch_printing_counter.remote((corrected_counter[batch_len: batch_len + batch_size])) for batch_len in range(0, len(corrected_counter), batch_size)])
        # print("Solids counter :")
        # ray.get([batch_printing.remote((solids[batch_len: batch_len + batch_size])) for batch_len in range(0, len(solids), batch_size)])
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

