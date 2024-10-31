from numba import cuda
from helpers import *

@cuda.jit
def two_sided_kernel(kmer_spectrum, reads, offsets, result, kmer_len):
    threadIdx = cuda.grid(1)

    #if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if threadIdx < offsets.shape[0]:

        #find the read assigned to this thread
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        MAX_LEN = 300
        bases = cuda.local.array(4, dtype='uint8')
        solids = cuda.local.array(MAX_LEN, dtype='int8')

        for i in range(end - start):
            solids[i] = -1

        for i in range(4):
            bases[i] = i + 1

        #identify whether base is solid or not
        identify_solid_bases(reads, start, end, kmer_len, kmer_spectrum, solids)

        #used for debugging
        for idx in range(end - start):
            result[threadIdx][idx] = solids[idx]

        #check whether base is potential for correction
        #kulang pani diria sa pag check sa first and last bases
        for base_idx in range(end - start):
            #the base needs to be corrected
            if solids[base_idx] == -1 and base_idx >= (kmer_len - 1) and base_idx <= (end - start) - kmer_len:

                left_portion = reads[(base_idx + start) - (kmer_len - 1): base_idx + start + 1]
                right_portion = reads[base_idx + start: base_idx + kmer_len]

                correct_reads(base_idx + start, reads, kmer_len, kmer_spectrum, bases, left_portion, right_portion)

            #the leftmost bases of the read
            if solids[base_idx] == -1 and base_idx < (kmer_len - 1):
                pass
            #the rightmost bases of the read
            if solids[base_idx] == -1 and base_idx > (end - start) - kmer_len:
                pass

#no implementation for tracking how many corrections are done for each kmers in the read
@cuda.jit
def one_sided_kernel(kmer_spectrum, reads_1d, offsets, kmer_len, solids_counter, solids_after, not_corrected_counter):
    threadIdx = cuda.grid(1)

    if threadIdx < offsets.shape[0]:

        MAX_LEN = 300
        region_indices = cuda.local.array((10,2), dtype="int32")
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
                while region_end != (end - start) - 1:
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

#for correcting reads in two-sided
@cuda.jit(device=True)
def correct_reads(idx, reads, kmer_len, kmer_spectrum,  bases, left_kmer, right_kmer ):
    current_base = reads[idx]
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
        reads[idx] = candidate
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    if posibility == 0:
        pass
        # corrected_counter[threadIdx][counter] = 10
        # counter += 1

    #ignore the correction if more than one possibility
    if  posibility > 1:
        pass
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    # return counter
