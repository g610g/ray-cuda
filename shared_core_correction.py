from numba import cuda
from shared_helpers import (
    identify_solid_bases,
    identify_trusted_regions,
    predeccessor,
    successor,
)
from helpers import in_spectrum, transform_to_key, give_kmer_multiplicity
from voting import apply_vm_result, invoke_voting


@cuda.jit
def two_sided_kernel(kmer_spectrum, reads, offsets, kmer_len):
    threadIdx = cuda.grid(1)

    # if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if threadIdx < offsets.shape[0]:

        # find the read assigned to this thread
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        MAX_LEN = 300
        bases = cuda.local.array(4, dtype="uint8")
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        local_reads = cuda.local.array(300, dtype="uint8")
        rpossible_base_mutations = cuda.local.array(10,dtype='uint8')
        lpossible_base_mutations = cuda.local.array(10,dtype='uint8')
        size = (end - start)  - kmer_len
        max_iters = 2
        # we try to transfer the reads assigned for this thread into its private memory for memory access issues
        for idx in range(0, end - start):
            local_reads[idx] = reads[idx + start]
        for i in range(4):
            bases[i] = i + 1

        for _ in range(max_iters):
            num_corrections = correct_two_sided(end, start, kmer_spectrum, kmer_len, bases, solids, local_reads, lpossible_base_mutations, rpossible_base_mutations, size)
            
            #this read is error free. Stop correction
            if num_corrections == 0:
                return
            if num_corrections < 0:
                break
        #bring local read back to global memory reads
        for idx in range(end - start):
            reads[idx + start] = local_reads[idx]

@cuda.jit(device=True)
def correct_two_sided(end, start, kmer_spectrum, kmer_len, bases, solids, local_read, lpossible_base_mutations, rpossible_base_mutations, size):
    for i in range(end - start):
        solids[i] = -1

    # identify whether base is solid or not
    identify_solid_bases(
        local_read, start, end, kmer_len, kmer_spectrum, solids
    )
    
    #check whether solids array does not contain -1, return 0 if yes

    klen_idx = kmer_len - 1
    for ipos in range(0, size + 1):
        lpos = 0
        if ipos >= 0:
            rkmer = local_read[ipos: ipos + kmer_len] 
        if ipos >= kmer_len:
            lkmer = local_read[ipos - klen_idx: ipos + 1]
            lpos = -1
        else: 
            lkmer = local_read[0: kmer_len]
            lpos = ipos
        #trusted base
        if solids[ipos] == 1:
            continue

        #do corrections right here
        
        #select all possible mutations for rkmer
        rnum_bases = 0
        for base in bases:
            rkmer[0] = base
            candidate_kmer = transform_to_key(rkmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                rpossible_base_mutations[rnum_bases] = base
                rnum_bases += 1 
            
        #select all possible mutations for lkmer
        lnum_bases = 0
        for base in bases:
            lkmer[lpos] = base
            candidate_kmer = transform_to_key(lkmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                lpossible_base_mutations[lnum_bases] = base
                lnum_bases += 1

        i = 0
        num_corrections = 0
        potential_base = -1
        while(i < rnum_bases and num_corrections <= 1):
            rbase = rpossible_base_mutations[i]
            j = 0 
            while (j < lnum_bases):
                lbase = lpossible_base_mutations[j]
                #add the potential correction
                if lbase == rbase:
                    num_corrections += 1
                    potential_base = rbase
                j += 1
            i += 1
        #apply correction to the current base and return from this function
        if num_corrections == 1 and potential_base != -1:
            local_read[ipos] = potential_base
            return 1
        
        #two sided stops if num corrections != 1
        # else:
        #     break
    
    #endfor  0 < seqlen - klen
    
    #for bases > (end - start) - klen)

    
    for ipos in range(size + 1, end - start):
        rkmer = local_read[size:]
        lkmer = local_read[ipos - klen_idx: ipos + 1]
        if solids[ipos] == 1:
            continue
        #select mutations for right kmer
        rnum_bases  = 0
        for base in bases:
            rkmer[ipos - size] = base
            candidate_kmer = transform_to_key(rkmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                rpossible_base_mutations[rnum_bases] = base
                rnum_bases += 1
        lnum_bases = 0
        for base in bases:
            lkmer[-1] = base
            candidate_kmer = transform_to_key(lkmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                lpossible_base_mutations[lnum_bases] = base
                lnum_bases += 1
        i = 0
        num_corrections = 0
        potential_base = -1
        while(i < rnum_bases and num_corrections <= 1):
            rbase = rpossible_base_mutations[i]
            j = 0 
            while (j < lnum_bases):
                lbase = lpossible_base_mutations[j]
                #add the potential correction
                if lbase == rbase:
                    num_corrections += 1
                    potential_base = rbase
                j += 1
            i += 1
        #apply correction to the current base and return from this function
        if num_corrections == 1 and potential_base != -1:
            local_read[ipos] = potential_base
            return 1
    return -1

@cuda.jit()
def one_sided_kernel(
    kmer_spectrum,
    reads,
    offsets,
    kmer_len,
    not_corrected_counter,
):
    threadIdx = cuda.grid(1)
    if threadIdx < offsets.shape[0]:

        MAX_LEN = 300

        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        alternatives = cuda.local.array((4, 2), dtype="uint32")
        corrected_solids = cuda.local.array(MAX_LEN, dtype="int8")
        region_indices = cuda.local.array((10, 2), dtype="int8")
        local_read = cuda.local.array(MAX_LEN, dtype="uint8")
        correction_tracker = cuda.local.array(MAX_LEN, dtype="uint8")
        original_read = cuda.local.array(MAX_LEN, dtype="uint8")
        voting_matrix = cuda.local.array((4, MAX_LEN), dtype="uint16")
        # number of kmers generated base on the length of reads and kmer
        num_kmers = (end - start) - (kmer_len - 1)
        maxIters = 4

        bases = cuda.local.array(4, dtype="uint8")

        # seeding bases 1 to 4
        for i in range(4):
            bases[i] = i + 1

        # transfer global memory store reads to local thread memory read
        for idx in range(end - start):
            local_read[idx] = reads[idx + start]
            original_read[idx] = reads[idx + start]

        for max_correction in range(1, maxIters + 1):
            for _ in range(2):
                for i in range(end - start):
                    solids[i] = -1

                # used for debugging
                for i in range(end - start):
                    corrected_solids[i] = -1

                # identifies trusted regions in this read
                regions_count = identify_trusted_regions(
                    start, end, kmer_spectrum, local_read, kmer_len, region_indices, solids
                )

                # zero regions count means no trusted region in this read
                if regions_count == 0:
                    return
                # for reads that has no error 
                if regions_count == 1 and region_indices[0][0] == 0 and region_indices[0][1] == 99:
                    return

                for region in range(regions_count):
                    # 1. goes toward right orientation

                    # there is no next region
                    if region == (regions_count - 1):
                        region_end = region_indices[region][1]
                        original_region_end = region_end
                        corrections_count = 0
                        last_position = -1
                        original_bases = local_read[region_end + 1 :]
                        # while we are not at the end base of the read
                        while region_end != (end - start) - 1:
                            target_pos = region_end + 1
                            # keep track the number of corrections made here
                            if not correct_read_one_sided_right(
                                local_read,
                                region_end,
                                kmer_spectrum,
                                kmer_len,
                                bases,
                                alternatives,
                            ):
                                not_corrected_counter[threadIdx] += 1
                                break

                            # extend the portion of region end for successful correction and check the number of corrections done
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if target_pos - last_position < kmer_len:
                                    corrections_count += 1

                                    # revert back bases from last position to target position and revert region
                                    # prevents cumulative incorrect corrections
                                    if corrections_count > max_correction:
                                        for pos in range(last_position, target_pos + 1):
                                            local_read[pos] = original_bases[
                                                pos - last_position
                                            ]
                                        region_indices[region][1] = original_region_end
                                        break
                                    region_end += 1
                                    region_indices[region][1] = target_pos
                                # recalculate the last position 
                                else:
                                    last_position = target_pos
                                    original_region_end = target_pos
                                    corrections_count = 0
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                    # there is a next region
                    elif region != (regions_count - 1):
                        region_end = region_indices[region][1]
                        original_region_end = region_end
                        next_region_start = region_indices[region + 1][0]
                        original_bases = local_read[region_end + 1 :]
                        last_position = -1
                        corrections_count = 0
                        # the loop will not stop until it does not find another region
                        while region_end != (next_region_start - 1):
                            target_pos = region_end + 1
                            if not correct_read_one_sided_right(
                                local_read,
                                region_end,
                                kmer_spectrum,
                                kmer_len,
                                bases,
                                alternatives,
                            ):
                                # fails to correct this region and on this orientation
                                not_corrected_counter[threadIdx] += 1
                                break

                            # extend the portion of region end for successful correction
                            else:

                                if last_position < 0:
                                    last_position = target_pos
                                if target_pos - last_position < kmer_len:
                                    corrections_count += 1

                                    # revert back bases from last position to target position and stop correction for this region oritentation
                                    if corrections_count > max_correction:
                                        for pos in range(last_position, target_pos + 1):
                                            local_read[pos] = original_bases[
                                                pos - last_position
                                            ]
                                        region_indices[region][1] = original_region_end
                                        break
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                                # recalculate the last position and reset corrections made
                                else:
                                    last_position = target_pos
                                    original_region_end = target_pos
                                    corrections_count = 0
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                    # 2. Orientation going to the left 
                    # leftmost region
                    if region == 0:
                        region_start = region_indices[region][0]
                        original_region_start = region_start
                        last_position = -1
                        corrections_count = 0
                        original_bases = local_read[0 : region_start + 1]

                        # while we are not at the first base of the read
                        while region_start > 0:
                            target_pos = region_start - 1
                            if not correct_read_one_sided_left(
                                local_read,
                                region_start,
                                kmer_spectrum,
                                kmer_len,
                                bases,
                                alternatives,
                            ):
                                not_corrected_counter[threadIdx] += 1
                                break
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if last_position - target_pos < kmer_len:
                                    corrections_count += 1
                                    # revert back bases
                                    if corrections_count > max_correction:
                                        for pos in range(target_pos, last_position + 1):
                                            local_read[pos] = original_bases[pos]
                                        region_indices[region][0] = original_region_start
                                        break
                                    region_start -= 1
                                    region_indices[region][0] = region_start
                                else:
                                    last_position = target_pos
                                    original_region_start = target_pos
                                    corrections_count = 0
                                    region_start = target_pos
                                    region_indices[region][0] = region_start

                    # there is another region in the left side of this region
                    elif region > 0:
                        region_start, prev_region_end = (
                            region_indices[region][0],
                            region_indices[region - 1][1],
                        )
                        original_region_start = region_start
                        last_position = -1
                        corrections_count = 0
                        original_bases = local_read[0 : region_start + 1]

                        while region_start - 1 > (prev_region_end):
                            target_pos = region_start - 1
                            if not correct_read_one_sided_left(
                                local_read,
                                region_start,
                                kmer_spectrum,
                                kmer_len,
                                bases,
                                alternatives,
                            ):
                                not_corrected_counter[threadIdx] += 1
                                break
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if last_position - target_pos < kmer_len:
                                    corrections_count += 1
                                    # revert back bases
                                    if corrections_count > max_correction:
                                        for pos in range(target_pos, last_position + 1):
                                            local_read[pos] = original_bases[pos]
                                        region_indices[region][0] = original_region_start
                                        break
                                    region_start -= 1
                                    region_indices[region][0] = region_start
                                else:
                                    last_position = target_pos
                                    original_region_start = target_pos
                                    corrections_count = 0
                                    region_start -= 1
                                    region_indices[region][0] = region_start

            voting_end_idx = (end - start) - kmer_len
            for idx in range(0, voting_end_idx):
                # start of the voting based refinement(havent integrated tracking kmer during apply_vm_result function)
                # for idx in range(start, end - (kmer_len - 1)):
                ascii_curr_kmer = local_read[idx : idx + kmer_len]
                curr_kmer = transform_to_key(ascii_curr_kmer, kmer_len)
            
                # invoke voting if the kmer is not in spectrum
                if not in_spectrum(kmer_spectrum, curr_kmer):
                    invoke_voting(
                        voting_matrix,
                        kmer_spectrum,
                        bases,
                        kmer_len,
                        idx,
                        local_read,
                    )
            # apply the result of the vm into the reads
            apply_vm_result(voting_matrix, local_read, start, end)
        # endfor idx to max_corrections

        # copies back corrected local read into global memory stored reads
        for idx in range(end - start):
            reads[idx + start] = local_read[idx]


# for orientation going to the right of the read
@cuda.jit(device=True)
def correct_read_one_sided_right(
    local_read,
    region_end,
    kmer_spectrum,
    kmer_len,
    bases,
    alternatives,
):

    possibility = 0
    alternative = -1
    target_pos = region_end + 1
    ipos = target_pos - (kmer_len - 1)

    forward_kmer = local_read[ipos : target_pos + 1]

    # foreach alternative base
    for alternative_base in bases:
        forward_kmer[-1] = alternative_base
        candidate_kmer = transform_to_key(forward_kmer, kmer_len)

        # if the candidate kmer is in the spectrum and has addition evidence that alternative base in trusted as correction by assessing neighbor kmers
        if in_spectrum(kmer_spectrum, candidate_kmer):

            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )
            possibility += 1
            alternative = alternative_base

    # returning false will should cause the caller to break the loop since it fails to correct (base on the Musket paper)
    if possibility == 0:
        return False

    # not sure if correct indexing for reads
    if possibility == 1:

        local_read[target_pos] = alternative
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:

        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            is_potential_correction = successor(
                kmer_len,
                local_read,
                kmer_spectrum,
                alternatives[idx][0],
                2,
                ipos,
            )
            if is_potential_correction:
                if alternatives[idx][1] >= choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            return True
        return False


# for orientation going to the left of the read
@cuda.jit(device=True)
def correct_read_one_sided_left(
    local_read,
    region_start,
    kmer_spectrum,
    kmer_len,
    bases,
    alternatives,
):

    possibility = 0
    alternative = -1

    target_pos = region_start - 1
    backward_kmer = local_read[target_pos : target_pos + kmer_len]
    # If end kmer of trusted region is at the spectrum and when sliding the window, the result kmer is not trusted, then we assume that the end base of that kmer is the sequencing error
    for alternative_base in bases:
        backward_kmer[0] = alternative_base
        candidate_kmer = transform_to_key(backward_kmer, kmer_len)

        # if the candidate kmer is in the spectrum and has addition evidence that alternative base in trusted as correction by assessing neighbor kmers
        if in_spectrum(kmer_spectrum, candidate_kmer):
            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )
            possibility += 1
            alternative = alternative_base

    # returning false should cause the caller to break the loop since it fails to correct (base on the Musket paper)
    if possibility == 0:
        return False

    # not sure if correct indexing for reads
    if possibility == 1:

        local_read[target_pos] = alternative
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:

        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            is_potential_correction = predeccessor(
                kmer_len,
                local_read,
                kmer_spectrum,
                target_pos,
                alternatives[idx][0],
                1,
            )
            if is_potential_correction:
                if alternatives[idx][1] >= choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            return True
        return False

