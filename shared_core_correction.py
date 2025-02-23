from numba import cuda
from numpy import copy
from shared_helpers import (
    all_solid_base,
    complement,
    identify_solid_bases,
    forward_base,
    identify_trusted_regions,
    lower,
    predeccessor,
    predeccessor_v2,
    reverse_comp,
    successor,
    successor_v2,
    copy_kmer,
    select_mutations,
    to_decimal_ascii,
)
from helpers import in_spectrum, transform_to_key, give_kmer_multiplicity


@cuda.jit(device=True)
def cast_votes(
    local_read, vm, seq_len, kmer_len, bases, size, kmer_spectrum, ascii_kmer, rep, aux_kmer, aux_km2
):
    # reset voting matrix
    for i in range(seq_len):
        for j in range(4):
            vm[i][j] = 0
    max_vote = 0

    # check each kmer within the read (planning to put the checking of max vote within this iteration)
    for ipos in range(0, size + 1):
        rev_comp = True
        copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
        copy_kmer(rep, local_read, ipos, ipos + kmer_len)
        reverse_comp(rep, kmer_len) 

        if lower(rep, ascii_kmer):
            copy_kmer(rep, ascii_kmer, 0, kmer_len)
            rev_comp = False

        kmer = transform_to_key(rep, kmer_len)
        if in_spectrum(kmer_spectrum, kmer):
            continue

        for base_idx in range(kmer_len):

            idx = base_idx
            if rev_comp:
                idx = kmer_len - base_idx - 1

            original_base = rep[idx]

            for base in bases:
                if original_base == base:
                    continue
                copy_kmer(aux_kmer, rep, 0, kmer_len)
                aux_kmer[idx] = base
                copy_kmer(aux_km2, aux_kmer, 0, kmer_len)
                reverse_comp(aux_km2, kmer_len)

                if lower(aux_km2, aux_kmer):
                    copy_kmer(aux_km2, aux_kmer, 0, kmer_len)

                candidate = transform_to_key(aux_km2 , kmer_len)
                if in_spectrum(kmer_spectrum, candidate):
                    if rev_comp:
                        base = complement(base)

                    vm[ipos + base_idx][base - 1] += 1
            ascii_kmer[base_idx] = original_base

    # find maximum vote
    for ipos in range(0, seq_len):
        for idx in range(4):
            if vm[ipos][idx] >= max_vote:
                max_vote = vm[ipos][idx]

    return max_vote


@cuda.jit(device=True)
def apply_voting_result(local_read, vm, seq_len, bases, max_vote):
    for ipos in range(seq_len):
        alternative_base = -1
        for base_idx in range(4):
            if vm[ipos][base_idx] == max_vote:
                if alternative_base == -1:
                    alternative_base = base_idx + 1
                else:
                    alternative_base = -1
        # apply the base correction if we have found an alternative base
        if alternative_base > 0:
            local_read[ipos] = alternative_base

@cuda.jit
def two_sided_kernel(kmer_spectrum, reads, offsets, kmer_len):
    threadIdx = cuda.grid(1)

    # if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
    if threadIdx < offsets.shape[0]:

        # find the read assigned to this thread
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        MAX_LEN = 300
        KMER_LEN = 21
        bases = cuda.local.array(4, dtype="uint8")
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        local_reads = cuda.local.array(MAX_LEN, dtype="uint8")
        rpossible_base_mutations = cuda.local.array((3, 2), dtype="uint8")
        lpossible_base_mutations = cuda.local.array((3, 2), dtype="uint8")
        aux_kmer = cuda.local.array(KMER_LEN, dtype="uint8")
        ascii_kmer = cuda.local.array(KMER_LEN, dtype="uint8")
        aux_km2 = cuda.local.array(KMER_LEN, dtype="uint8")
        rep = cuda.local.array(KMER_LEN, dtype="uint8")

        seqlen = end - start
        size = seqlen - kmer_len

        # this should be a terminal argument
        max_iters = 2
        # we try to transfer the reads assigned for this thread into its private memory for memory access issues
        for idx in range(0, end - start):
            local_reads[idx] = reads[idx + start]

        for i in range(4):
            bases[i] = i + 1

        for _ in range(max_iters):
            num_corrections = correct_two_sided(
                seqlen,
                kmer_spectrum,
                ascii_kmer,
                aux_kmer,
                kmer_len,
                bases,
                solids,
                local_reads,
                lpossible_base_mutations,
                rpossible_base_mutations,
                size,
                rep,
                aux_km2,
            )
            # this read is error free. Stop correction
            if num_corrections == 0:
                break

            # this read has more one than error within a kmer. Pass the read to one sided correction
            if num_corrections < 0:
                break

        # bring local read back to global memory reads
        for idx in range(end - start):
            reads[idx + start] = local_reads[idx]


@cuda.jit(device=True)
def correct_two_sided(
    seqlen,
    kmer_spectrum,
    ascii_kmer,
    aux_kmer,
    kmer_len,
    bases,
    solids,
    local_read,
    lpossible_base_mutations,
    rpossible_base_mutations,
    size,
    rep,
    aux_km2
):
    for i in range(seqlen):
        solids[i] = -1

    # identify whether base is solid or not
    identify_solid_bases(local_read, kmer_len, kmer_spectrum, solids, ascii_kmer, size, aux_kmer)
    # check whether solids array does not contain -1, return 0 if yes
    if all_solid_base(solids, seqlen):
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
        # trusted base
        if solids[ipos] == 1:
            continue

        #checks for reverse complement and checking lowest canonical representation 
        copy_kmer(rep, ascii_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, ascii_kmer):
            copy_kmer(rep, ascii_kmer, 0, kmer_len)
            rev_comp = False

        # select all possible mutation right kmer
        rnum_bases = select_mutations(
            kmer_spectrum, bases, rep, kmer_len, 0, rpossible_base_mutations, rev_comp, ascii_kmer, aux_km2
        )
        #checks for reverse complement and checking lowest canonical representation 
        copy_kmer(rep, aux_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, aux_kmer):
            copy_kmer(rep, aux_kmer, 0, kmer_len)
            rev_comp = False

        # select all possible mutations for left kmer
        lnum_bases = select_mutations(
            kmer_spectrum, bases, rep, kmer_len, lpos, lpossible_base_mutations, rev_comp, aux_kmer, aux_km2
        )

        i = 0
        num_corrections = 0
        potential_base = -1
        while i < rnum_bases and num_corrections <= 1:
            rbase = rpossible_base_mutations[i][0]
            j = 0
            while j < lnum_bases:
                lbase = lpossible_base_mutations[j][0]
                # add the potential correction
                if lbase == rbase:
                    num_corrections += 1
                    potential_base = rbase
                j += 1
            i += 1
        # apply correction to the current base and return from this function
        if num_corrections == 1 and potential_base > 0:
            local_read[ipos] = potential_base
            return 1

    # endfor  0 < seqlen - klen

    # for bases > seqlen - klen)

    for ipos in range(size + 1, seqlen):
        copy_kmer(ascii_kmer, local_read, size, seqlen)
        copy_kmer(aux_kmer, local_read, ipos - klen_idx, ipos + 1)
        if solids[ipos] == 1:
            continue
        copy_kmer(rep, ascii_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, ascii_kmer):
            copy_kmer(rep, ascii_kmer, 0, kmer_len)
            rev_comp  = False

        # select mutations for right kmer
        rnum_bases = select_mutations(
            kmer_spectrum,
            bases,
            rep,
            kmer_len,
            ipos - size,
            rpossible_base_mutations,
            rev_comp,
            ascii_kmer,
            aux_km2
        )

        copy_kmer(rep, aux_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, aux_kmer):
            copy_kmer(rep, aux_kmer, 0, kmer_len)
        else:
            rev_comp = False

        lnum_bases = select_mutations(
            kmer_spectrum, bases, rep, kmer_len, klen_idx, lpossible_base_mutations, rev_comp, aux_kmer, aux_km2
        )
        i = 0
        num_corrections = 0
        potential_base = -1
        while i < rnum_bases and num_corrections <= 1:
            rbase = rpossible_base_mutations[i][0]
            j = 0
            while j < lnum_bases:
                lbase = lpossible_base_mutations[j][0]
                # add the potential correction
                if lbase == rbase:
                    num_corrections += 1
                    potential_base = rbase
                j += 1
            i += 1
        # apply correction to the current base and return from this function
        if num_corrections == 1 and potential_base > 0:
            local_read[ipos] = potential_base
            return 1
    return -1


@cuda.jit()
def one_sided_kernel(kmer_spectrum, reads, offsets, kmer_len, max_votes):
    threadIdx = cuda.grid(1)
    if threadIdx < offsets.shape[0]:

        MAX_LEN = 300
        DEFAULT_KMER_LEN = 21
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        region_indices = cuda.local.array((20, 2), dtype="int32")
        voting_matrix = cuda.local.array((MAX_LEN, 4), dtype="uint32")
        selected_bases = cuda.local.array((3, 2), dtype="uint8")
        km = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        aux_km = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        aux_km2 = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        rep = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        bases = cuda.local.array(4, dtype="uint8")
        local_read = cuda.local.array(MAX_LEN, dtype="uint8")
        aux_corrections = cuda.local.array(MAX_LEN, dtype="uint8")
        local_read_aux = cuda.local.array(MAX_LEN, dtype="uint8")
        maxIters = 6
        min_vote = 3
        seqlen = end - start
        early_exit = False

        # seeding bases 1 to 4
        for i in range(0, 4):
            bases[i] = i + 1

        # transfer global memory store reads to local thread memory read
        for idx in range(end - start):
            local_read[idx] = reads[start + idx]

        for nerr in range(1, maxIters + 1):
            if early_exit:
                break

            distance = maxIters - nerr + 1
            for _ in range(2):
                # reset solids and aux_corrections every before run of onesided
                for idx in range(seqlen):
                    solids[idx] = -1
                    aux_corrections[idx] = 0
                    local_read_aux[idx] = local_read[idx]

                corrections_made = one_sided_v2(
                    local_read,
                    aux_corrections,
                    km,
                    aux_km,
                    region_indices,
                    selected_bases,
                    kmer_len,
                    seqlen,
                    kmer_spectrum,
                    solids,
                    bases,
                    nerr,
                    distance,
                    local_read_aux,
                    rep,
                    aux_km2
                )

                # returns -1 means all bases are solid
                if corrections_made == -1:
                    early_exit = True
                    break
                # no corrections made
                if corrections_made == 0:
                    break

                # a correction has been recorded and put the corrections in local read
                else:
                    for i in range(seqlen):
                        if aux_corrections[i]  > 0:
                            local_read[i] = aux_corrections[i]

            # start voting refinement here

            max_vote = cast_votes(
                local_read,
                voting_matrix,
                seqlen,
                kmer_len,
                bases,
                seqlen - kmer_len,
                kmer_spectrum,
                km,
                rep,
                aux_km,
                aux_km2
            )

            max_votes[threadIdx][nerr - 1] = max_vote

            #the read is error free at this point
            if max_vote == 0:
                break
            elif max_vote >= min_vote:
                apply_voting_result(local_read, voting_matrix, seqlen, bases, max_vote)
        # endfor idx to max_corrections

        # copies back corrected local read into global memory stored reads
        for idx in range(seqlen):
            reads[start + idx] = local_read[idx]


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
def one_sided_v2(
    original_read,
    aux_corrections,
    ascii_kmer,
    aux_kmer,
    region_indices,
    selected_bases,
    kmer_len,
    seq_len,
    spectrum,
    solids,
    bases,
    max_corrections,
    distance,
    local_read_aux,
    rep,
    aux_km2
):

    corrections_made = 0
    size = seq_len - kmer_len
    regions_count = identify_trusted_regions(
        seq_len,
        spectrum,
        original_read,
        kmer_len,
        region_indices,
        solids,
        aux_kmer,
        size,
        rep
    )

    if all_solid_base(solids, seq_len):
        return -1

    for region in range(0, regions_count):
        right_mer_idx = region_indices[region][1]
        right_orientation_idx = -1
        last_position = -1
        num_corrections = 0
        for target_pos in range(right_mer_idx + 1, seq_len):
            if solids[target_pos] == 1:
                break

            done = False
            spos = target_pos - (kmer_len - 1)

            if target_pos == (right_mer_idx + 1):
                copy_kmer(ascii_kmer, original_read, spos, target_pos + 1)
            else:
                forward_base(ascii_kmer, original_read[target_pos], kmer_len)

            #checks for reverse complement and checking lowest canonical representation 
            copy_kmer(rep, ascii_kmer, 0, kmer_len)
            reverse_comp(rep, kmer_len)
            rev_comp = True

            if lower(rep, ascii_kmer):
                copy_kmer(rep, ascii_kmer, 0, kmer_len)
                rev_comp = False

            # select all possible mutations

            num_bases = select_mutations(
                spectrum, bases, rep, kmer_len, kmer_len - 1, selected_bases, rev_comp, aux_kmer, aux_km2
            )

            # directly apply correction if mutation is equal to 1
            if num_bases == 1:
                if right_orientation_idx == -1:
                    right_orientation_idx = target_pos
                aux_corrections[target_pos] = selected_bases[0][0]
                ascii_kmer[kmer_len - 1] = selected_bases[0][0]
                corrections_made += 1
                done = True
            else:
                best_base = -1
                best_base_occurence = -1
                for idx in range(0, num_bases):
                    copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                    if successor_v2(
                        kmer_len,
                        original_read,
                        aux_kmer,
                        spectrum,
                        selected_bases[idx][0],
                        spos,
                        distance,
                        seq_len,
                        rep,
                    ):
                        if selected_bases[idx][1] > best_base_occurence:
                            best_base_occurence = selected_bases[idx][1]
                            best_base = selected_bases[idx][0]

                if best_base_occurence > 0 and best_base > 0:
                    if right_orientation_idx == -1:
                        right_orientation_idx = target_pos

                    aux_corrections[target_pos] = best_base
                    ascii_kmer[kmer_len - 1] = best_base
                    corrections_made += 1
                    done = True

            # check how many corrections is done and extend the region index
            if done:
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    # revert back reads if corrections exceed max_corrections
                    if num_corrections > max_corrections:
                        for pos in range(last_position, target_pos + 1):
                            aux_corrections[pos] = 0

                        corrections_made -= num_corrections
                        break
                        # break correction for this orientation after reverting back kmers
                else:
                    last_position = target_pos
                    num_corrections = 0

                continue

            # break correction if done is False
            break

        # endfor rkmer_idx + 1 to seq_len

        # for left orientation
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:

            copy_kmer(local_read_aux, original_read, 0, seq_len)
            last_position = right_orientation_idx

            num_corrections = 0
            pos = lkmer_idx - 1
            while pos >= 0:
                # the current base is trusted
                if solids[pos] == 1:
                    break

                done = False

                if pos == (lkmer_idx - 1):
                    copy_kmer(ascii_kmer, original_read, pos, pos + kmer_len)
                else:
                    copy_kmer(ascii_kmer, local_read_aux, pos, pos + kmer_len)

                rev_comp = True
                copy_kmer(rep, ascii_kmer, 0, kmer_len)
                reverse_comp(rep, kmer_len)

                if lower(rep, ascii_kmer):
                    copy_kmer(rep, ascii_kmer, 0, kmer_len)
                    rev_comp = False

                kmer = transform_to_key(rep, kmer_len)
                if not in_spectrum(spectrum, kmer):
                    num_bases = select_mutations(
                        spectrum, bases, rep, kmer_len, 0, selected_bases,rev_comp,  aux_kmer, aux_km2
                    )

                    # apply correction
                    if num_bases == 1:
                        aux_corrections[pos] = selected_bases[0][0]
                        local_read_aux[pos] = selected_bases[0][0]
                        corrections_made += 1
                        done = True

                    else:
                        best_base = -1
                        best_base_occurence = -1
                        for idx in range(0, num_bases):
                            copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                            if predeccessor_v2(
                                kmer_len,
                                local_read_aux,
                                aux_kmer,
                                spectrum,
                                pos,
                                selected_bases[idx][0],
                                distance,
                                rep,
                            ):
                                if  selected_bases[idx][1] > best_base_occurence:
                                    best_base_occurence = selected_bases[idx][1] 
                                    best_base = selected_bases[idx][0]

                        if best_base > 0 and best_base_occurence > 0:
                            aux_corrections[pos] = best_base
                            local_read_aux[pos] = best_base
                            corrections_made += 1
                            done = True
                    # checking corrections that have done
                    if done:
                        if last_position < 0:
                            last_position = pos
                        if last_position - pos < kmer_len:
                            num_corrections += 1

                            # revert kmer back if corrections done exceeds max_corrections
                            if num_corrections > max_corrections:
                                for base_idx in range(pos, last_position + 1):
                                    aux_corrections[base_idx] = 0

                                corrections_made -= num_corrections
                                break
                        else:
                            last_position = pos
                            num_corrections = 0
                        pos -= 1
                        continue

                    # the correction for the current base is done == False
                    pos -= 1
                    break
                #endif not in_spectrum
                pos -= 1
        # endfor lkmer_idx to 0

    # endfor regions_count
    return corrections_made
