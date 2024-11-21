import cudf
from numba import cuda
from helpers import in_spectrum, transform_to_key, mark_solids_array
from Bio import Seq
import ray



@cuda.jit(device=True)
def identify_solid_bases(local_reads, start, end, kmer_len, kmer_spectrum, solids):

    for idx in range(0, (end - start) - (kmer_len - 1)):
        ascii_kmer = local_reads[idx:idx + kmer_len]

        curr_kmer = transform_to_key(ascii_kmer, kmer_len)

        #set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx , (idx + kmer_len))

@cuda.jit(device=True)
def identify_trusted_regions(start, end, kmer_spectrum, local_reads, kmer_len, region_indices, solids):

    identify_solid_bases(local_reads, start, end, kmer_len, kmer_spectrum, solids)

    current_indices_idx = 0
    base_count = 0
    region_start = 0
    region_end = 0

    #idx will be a relative index
    for idx in range(end - start):

        #a trusted region has been found. Append it into the identified regions
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
            base_count+=1

    #ending
    if base_count >= kmer_len:

        region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, region_end
        current_indices_idx += 1

    #this will be the length or the number of trusted regions
    return current_indices_idx

@cuda.jit(device=True)
def to_local_reads(reads_1d, local_reads, start, end):

    for idx in range(start, end):
        local_reads[idx - start] = reads_1d[idx]


@ray.remote(num_cpus=1)
def transform_back_to_row_reads(reads, batch_start, batch_end, offsets ):

    for batch_idx in range(batch_start, batch_end):
        start, end = offsets[batch_idx][0], offsets[batch_idx][1]

#from jagged array to 2 dimensional array

@ray.remote(num_cpus=1,num_gpus=1)
def back_to_sequence_helper(reads,offsets):

    offsets_df = cudf.DataFrame({"start":offsets[:,0], "end":offsets[:,1]})
    offsets_df['length'] = offsets_df['end'] - offsets_df['start']
    max_segment_length = offsets_df['length'].max()

    cuda.profile_start()
    start = cuda.event()
    end = cuda.event()
    start.record()

    dev_reads = cuda.to_device(reads)
    dev_offsets = cuda.to_device(offsets)
    dev_reads_result = cuda.device_array((offsets.shape[0], max_segment_length), dtype='uint8')
    tpb = 1024
    bpg = (offsets.shape[0] + tpb) // tpb

    back_sequence_kernel[bpg, tpb](dev_reads, dev_offsets , dev_reads_result)

    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the back to sequence kernel:  {transfer_time} ms")
    cuda.profile_stop()

    return dev_reads_result.copy_to_host()

@ray.remote(num_cpus=1)
def increment_array(arr):
    for value in arr:
        value += 1
    return arr

@ray.remote(num_cpus=1)
def assign_sequence(read_batch, sequence_batch):
    translation_table = str.maketrans({"1":"A", "2":"C", "3":"G", "4":"T", "5":"N"})
    for int_read, sequence in zip(read_batch, sequence_batch):
        non_zeros_int_read = [x for x in int_read if x != 0]
        read_string = ''.join(map(str, non_zeros_int_read)) 
        ascii_read_string = read_string.translate(translation_table)
        sequence.seq = Seq.Seq(ascii_read_string)
    return sequence_batch


@cuda.jit
def back_sequence_kernel(reads, offsets, reads_result):
    threadIdx = cuda.grid(1)
    MAX_LEN = 300
    local_reads = cuda.local.array(MAX_LEN, dtype='uint8')
    if threadIdx < offsets.shape[0]:
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        to_local_reads(reads, local_reads, start, end)

        #copy the assigned read for this thread into the 2d reads_result
        for idx in range(end - start):
            reads_result[threadIdx][idx] = local_reads[idx]

