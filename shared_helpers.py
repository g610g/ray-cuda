from numba import cuda
from helpers import in_spectrum, transform_to_key, mark_solids_array

@cuda.jit(device=True)
def identify_solid_bases(shared_reads, start, end, kmer_len, kmer_spectrum, solids, threadIdx_block):

    for idx in range(0, (end - start) - (kmer_len - 1)):
        curr_kmer = transform_to_key(shared_reads[threadIdx_block][idx:idx + kmer_len], kmer_len)

        #set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):

            mark_solids_array(solids, idx , (idx + kmer_len))

@cuda.jit(device=True)
def identify_trusted_regions(start, end, kmer_spectrum, shared_reads, kmer_len, region_indices, solids, threadIdx_block):

    identify_solid_bases(shared_reads, start, end, kmer_len, kmer_spectrum, solids, threadIdx_block)

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
