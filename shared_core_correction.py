from numpy import select
from numba import cuda
from shared_helpers import (
    identify_solid_bases,
    identify_trusted_regions,
    predeccessor,
    predeccessor_v2,
    successor,
    successor_v2,
    check_solids_cardinality,
    copy_kmer,
    select_mutations

)
from helpers import in_spectrum, transform_to_key, give_kmer_multiplicity


@cuda.jit(device=True)
def cast_votes(local_read, vm, seq_len, kmer_len, bases, size, kmer_spectrum, ascii_kmer):
    #reset voting matrix
    for i in range(seq_len):
        for j in range(len(bases)):
            vm[i][j] = 0
    max_vote = 0
    #check each kmer within the read (planning to put the checking of max vote within this iteration)
    for ipos in range(0, size + 1):

        copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
        kmer = transform_to_key(ascii_kmer, kmer_len)
        if in_spectrum(kmer_spectrum, kmer):
            continue
        for base_idx in range(kmer_len):
            original_base = ascii_kmer[base_idx]
            for base in bases:
                if original_base == base:
                    continue
                ascii_kmer[base_idx] = base
                candidate = transform_to_key(ascii_kmer, kmer_len)
                if in_spectrum(kmer_spectrum, candidate):
                    vm[ipos + base_idx][base - 1] += 1
            ascii_kmer[base_idx] = original_base

    #find maximum vote
    for ipos in range(0, seq_len):
        for idx in range(len(bases)):
           if vm[ipos][idx] >= max_vote:
               max_vote = vm[ipos][idx]

    return max_vote

@cuda.jit(device=True)
def apply_voting_result(local_read, vm, seq_len, bases, max_vote):
    for ipos in range(seq_len):
        alternative_base = -1
        for base_idx in range(len(bases)):
            if vm[ipos][base_idx] == max_vote:
                if alternative_base == -1:
                    alternative_base = base_idx + 1
                else:
                    alternative_base = -1
        #apply the base correction if we have found an alternative base
        if alternative_base >= 1:
            local_read[ipos] = alternative_base

@cuda.jit
def two_sided_kernel(kmer_spectrum, reads, offsets, kmer_len):
    threadIdx = cuda.grid(1)

    # if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if threadIdx < offsets.shape[0]:

        # find the read assigned to this thread
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        MAX_LEN = 300
        KMER_LEN = 13
        bases = cuda.local.array(4, dtype="uint8")
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        local_reads = cuda.local.array(300, dtype="uint8")
        rpossible_base_mutations = cuda.local.array(10,dtype='uint8')
        lpossible_base_mutations = cuda.local.array(10,dtype='uint8')
        aux_kmer = cuda.local.array(KMER_LEN, dtype='uint8')
        ascii_kmer = cuda.local.array(KMER_LEN, dtype='uint8')
        seqlen = end - start
        size = seqlen - kmer_len

        #this should be a terminal argument
        max_iters = 2
        # we try to transfer the reads assigned for this thread into its private memory for memory access issues
        for idx in range(0, end - start):
            local_reads[idx] = reads[idx + start]

        for i in range(4):
            bases[i] = i + 1

        for _ in range(max_iters):
            num_corrections = correct_two_sided(end, start, seqlen, kmer_spectrum, ascii_kmer, aux_kmer,  kmer_len, bases, solids, local_reads, lpossible_base_mutations, rpossible_base_mutations, size)
            #this read is error free. Stop correction
            if num_corrections == 0:
                return

            #this read has more one than error within a kmer. Pass the read to one sided correction
            if num_corrections < 0:
                break

        #bring local read back to global memory reads
        for idx in range(end - start):
            reads[idx + start] = local_reads[idx]

@cuda.jit(device=True)
def correct_two_sided(end, start, seqlen, kmer_spectrum, ascii_kmer, aux_kmer,  kmer_len, bases, solids, local_read, lpossible_base_mutations, rpossible_base_mutations, size):
    for i in range(end - start):
        solids[i] = -1

    # identify whether base is solid or not
    identify_solid_bases(
        local_read, kmer_len, kmer_spectrum, solids, ascii_kmer, size
    )
    #check whether solids array does not contain -1, return 0 if yes
    if check_solids_cardinality(solids, seqlen):
        return 0

    klen_idx = kmer_len - 1
    for ipos in range(0, size + 1):
        lpos = 0
        if ipos >= 0:
            copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
        if ipos >= kmer_len:
            copy_kmer(aux_kmer, local_read, ipos - klen_idx, ipos + 1)
            lpos = klen_idx
        else: 
            copy_kmer(aux_kmer, local_read, 0, kmer_len)
            lpos = ipos
        #trusted base
        if solids[ipos] == 1:
            continue

        #do corrections right here
        
        #select all possible mutations for rkmer
        rnum_bases = 0
        for base in bases:
            ascii_kmer[0] = base
            candidate_kmer = transform_to_key(ascii_kmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                rpossible_base_mutations[rnum_bases] = base
                rnum_bases += 1 
            
        #select all possible mutations for lkmer
        lnum_bases = 0
        for base in bases:
            aux_kmer[lpos] = base
            candidate_kmer = transform_to_key(aux_kmer, kmer_len)
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
        
    #endfor  0 < seqlen - klen
    
    #for bases > seqlen - klen)

    
    for ipos in range(size + 1, seqlen):
        copy_kmer(ascii_kmer, local_read, size, seqlen)
        copy_kmer(aux_kmer, local_read, ipos - klen_idx, ipos + 1)
        if solids[ipos] == 1:
            continue
        #select mutations for right kmer
        rnum_bases  = 0
        for base in bases:
            ascii_kmer[ipos - size] = base
            candidate_kmer = transform_to_key(ascii_kmer, kmer_len)
            if in_spectrum(kmer_spectrum, candidate_kmer):
                rpossible_base_mutations[rnum_bases] = base
                rnum_bases += 1
        lnum_bases = 0
        for base in bases:
            aux_kmer[klen_idx]= base
            candidate_kmer = transform_to_key(aux_kmer, kmer_len)
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
        DEFAULT_KMER_LEN = 13
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        region_indices = cuda.local.array((10, 2), dtype="int8")
        original_read = cuda.local.array(MAX_LEN, dtype="uint8")
        local_read = cuda.local.array(MAX_LEN, dtype="uint8")
        voting_matrix = cuda.local.array((MAX_LEN, 4), dtype="uint16")
        selected_bases = cuda.local.array(10 , dtype="uint8")
        aux_kmer = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        aux_kmer2 = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        bases = cuda.local.array(4, dtype="uint8")
      
        maxIters = 4
        min_vote = 3
        seqlen = end - start


        # seeding bases 1 to 4
        for i in range(4):
            bases[i] = i + 1

        # transfer global memory store reads to local thread memory read
        for idx in range(end - start):
            local_read[idx] = reads[idx + start]

        for nerr in range(1, maxIters + 1):
            max_correction = maxIters - nerr + 1
            for _ in range(2):
                
                #reset solids every before run of onesided
                for idx in range(seqlen):
                    solids[idx] = -1

                one_sided_v2(local_read, original_read, aux_kmer, aux_kmer2, region_indices, selected_bases, kmer_len, seqlen, kmer_spectrum, solids, bases, nerr, max_correction)
                        #start voting refinement here

            max_vote = cast_votes(local_read, voting_matrix, end - start, kmer_len, bases, (end - start) - kmer_len, kmer_spectrum,aux_kmer)
            
            #the read is error free at this point
            if max_vote == 0:
                return
            elif max_vote >= min_vote:
                apply_voting_result(local_read, voting_matrix, (end - start), bases, max_vote)
        
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

@cuda.jit(device=True)
def one_sided_v2(local_read, original_read, ascii_kmer, aux_kmer, region_indices, selected_bases, kmer_len, seq_len, spectrum, solids, bases, max_corrections, distance):

    size = seq_len - kmer_len
    regions_count = identify_trusted_regions(seq_len, spectrum, local_read, kmer_len, region_indices, solids, aux_kmer, size)

    #check -1 cardinality and return true if cardinality is == 0
    if check_solids_cardinality(solids, seq_len):
        return
    for region in range(regions_count):
        right_mer_idx = region_indices[region][1]
        last_position = -1
        num_corrections = 0

        #copy original read first
        copy_kmer(original_read, local_read, right_mer_idx + 1, seq_len)

        for target_pos in range(right_mer_idx + 1, seq_len):
            best_base = -1
            best_base_occurence = -1
            if solids[target_pos] == 1:
                break
            done = False
            spos = target_pos - (kmer_len - 1)

            #get the ascii kmer
            copy_kmer(ascii_kmer, local_read, spos, target_pos + 1)
            
            num_bases = select_mutations(spectrum, bases, ascii_kmer, kmer_len, kmer_len - 1, selected_bases)

            #directly apply correction if mutation is equal to 1
            if num_bases == 1:
                #print(f"correction toward right index {target_pos} has one alternative base: {alternative_bases[0]}")
                local_read[target_pos] = selected_bases[0]
                done = True
            else:
                #print(f"alternative bases: {alternative_bases}")
                for idx in range(0, num_bases):
                    if successor_v2(kmer_len, local_read, aux_kmer, spectrum, selected_bases[idx], spos, distance):
                        #print("successor returns true")
                        copy_kmer(aux_kmer, local_read, spos, target_pos + 1)
                        aux_kmer[kmer_len - 1] = selected_bases[idx]
                        candidate = transform_to_key(aux_kmer, kmer_len)
                        aux_occurence = give_kmer_multiplicity(spectrum, candidate)

                        if aux_occurence > best_base_occurence:
                            best_base_occurence = aux_occurence
                            best_base = selected_bases[idx]

                #apply correction
                if best_base_occurence != -1 and best_base != -1:
                    #print(f"{best_base} is the chosen alternative base")
                    local_read[target_pos] = best_base
                    done = True

            #check how many corrections is done and extend the region index
            if done:
                region_indices[region][1] = target_pos
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    #revert back reads if corrections exceed max_corrections
                    if num_corrections > max_corrections:
                        #print(f"Reverting corrections made from index: {last_position} to index: {target_pos}, max corrections is: {max_corrections} num corrections is :{num_corrections}")
                        for pos in range(last_position, target_pos + 1):
                            #print(f"Reverting base position: {pos} to {original_read[pos - (right_mer_idx + 1)]}")
                            local_read[pos] = original_read[pos - (right_mer_idx + 1)]
                        region_indices[region][1] =  last_position - 1
                        break
                        #break correction for this orientation after reverting back kmers
                else:
                    #modify original read elements right here
                    last_position = target_pos
                    copy_kmer(original_read, local_read, last_position, seq_len)
                    num_corrections = 0

                continue

            #break correction if done is False
            break

        #endfor rkmer_idx + 1 to seq_len

        #for left orientation
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:
            last_position = -1
            num_corrections = 0    

            #copy original read first
            copy_kmer(original_read, local_read, 0, lkmer_idx)

            for pos in range(lkmer_idx - 1, -1, -1):
                #the current base is trusted
                if solids[pos] == 1:
                    break
                best_base = -1
                best_base_occurence = -1
                done = False
                copy_kmer(ascii_kmer, local_read, pos, pos + kmer_len)
                num_bases = select_mutations(spectrum, bases, ascii_kmer, kmer_len, 0, selected_bases)

                # print(f"alternative bases for left orientation index from {pos} to {pos + (kmer_len - 1)} is {alternative_bases}")
                # apply correction 
                if num_bases == 1:
                    #print(f"correction toward left index {pos} has one alternative base: {alternative_bases[0]}")
                    local_read[pos] = selected_bases[0]
                    done = True
                else:
                    for idx in range(num_bases):
                        if predeccessor_v2(kmer_len, local_read, aux_kmer, spectrum, pos, selected_bases[idx], distance):
                            #print(f"predeccessor returns true for base {alternative_bases[idx]}")
                            copy_kmer(aux_kmer, local_read, pos, pos + kmer_len)
                            aux_kmer[0] = selected_bases[idx]
                            candidate = transform_to_key(aux_kmer, kmer_len)
                            aux_occurence = give_kmer_multiplicity(spectrum, candidate)
                            if aux_occurence > best_base_occurence:
                                best_base_occurence = aux_occurence
                                best_base = selected_bases[idx]

                    if best_base > -1 and best_base_occurence > 0:
                        local_read[pos] = best_base
                        done = True
                #checking corrections that have done
                if done:
                    region_indices[region][0] = pos
                    if last_position < 0:
                            last_position = pos
                    if last_position - pos < kmer_len:
                        num_corrections += 1

                        #revert kmer back if corrections done exceeds max_corrections
                        if num_corrections > max_corrections:
                            for base_idx in range(pos, last_position + 1):
                                local_read[base_idx] = original_read[base_idx]
                            region_indices[region][0] = last_position + 1
                            break
                    else:
                        last_position = pos
                        copy_kmer(original_read, local_read, 0, last_position + 1)
                        num_corrections = 0
                    continue

                #the correction for the current base is done == False
                break
            #endfor lkmer_idx to 0

    #endfor regions_count
