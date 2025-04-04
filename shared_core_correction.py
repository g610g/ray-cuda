from numba import cuda
from numba.cuda import target
from shared_helpers import (
    all_solid_base,
    complement,
    encode_bases,
    identify_solid_bases,
    identify_trusted_regions,
    identify_trusted_regions_v2,
    lower,
    predeccessor,
    predeccessor_v2,
    reverse_comp,
    sort_ping,
    sort_pong,
    successor,
    successor_v2,
    copy_kmer,
    select_mutations,
)
from helpers import in_spectrum, transform_to_key, give_kmer_multiplicity


@cuda.jit(device=True)
def cast_votes(
    local_read,
    vm,
    seq_len,
    kmer_len,
    bases,
    size,
    kmer_spectrum,
    km,
    rep,
    aux_kmer,
    aux_km2,
):
    # reset voting matrix
    for i in range(seq_len):
        for j in range(4):
            vm[i][j] = 0

    max_vote = 0
    error_free = True

    # check each kmer within the read (planning to put the checking of max vote within this iteration)
    for ipos in range(0, size + 1):
        rev_comp = True
        copy_kmer(km, local_read, ipos, ipos + kmer_len)
        copy_kmer(rep, km, 0, kmer_len)
        reverse_comp(rep, kmer_len)

        if lower(rep, km, kmer_len):
            copy_kmer(rep, km, 0, kmer_len)
            rev_comp = False

        kmer = transform_to_key(rep, kmer_len)
        if in_spectrum(kmer_spectrum, kmer):
            continue

        error_free = False

        # conduct voting on each position in the kmer
        for base_idx in range(0, kmer_len):

            idx = base_idx
            if rev_comp:
                idx = kmer_len - 1 - base_idx

            original_base = rep[idx]

            for i in range(4):
                if original_base == bases[i]:
                    continue
                copy_kmer(aux_kmer, rep, 0, kmer_len)
                aux_kmer[idx] = bases[i]

                copy_kmer(aux_km2, aux_kmer, 0, kmer_len)
                reverse_comp(aux_km2, kmer_len)

                if lower(aux_km2, aux_kmer, kmer_len):
                    copy_kmer(aux_km2, aux_kmer, 0, kmer_len)

                candidate = transform_to_key(aux_km2, kmer_len)
                if in_spectrum(kmer_spectrum, candidate):
                    base = bases[i]
                    if rev_comp:
                        base = complement(bases[i])

                    # add restriction range
                    if base > 0 and base < 5:
                        vm[ipos + base_idx][base - 1] += 1

    if error_free:
        return 0
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
        for idx in range(4):
            if vm[ipos][bases[idx] - 1] == max_vote:
                if alternative_base == -1:
                    alternative_base = bases[idx]
                else:
                    alternative_base = -1
        # apply the base correction if we have found an alternative base
        if alternative_base > 0 and alternative_base < 5:
            local_read[ipos] = alternative_base


# @cuda.jit
# def two_sided_kernel(
#     kmer_spectrum, reads, offsets, kmer_len, corrections_flag, last_end_idx, dev_solids, corrected_tracker
# ):
#     threadIdx = cuda.grid(1)
#
#     # if the rightside and leftside are present in the kmer spectrum, then assign 1 into the result. Otherwise, 0
#     if threadIdx < offsets.shape[0]:
#
#         # find the read assigned to this thread
#         start, end = offsets[threadIdx][0], offsets[threadIdx][1]
#         # start, end = initial_start - last_end_idx, initial_end - last_end_idx
#         MAX_LEN = 300
#         KMER_LEN = 19
#         bases = cuda.local.array(4, dtype="uint8")
#         solids = cuda.local.array(MAX_LEN, dtype="int8")
#         local_reads = cuda.local.array(MAX_LEN, dtype="uint8")
#         rpossible_base_mutations = cuda.local.array((4, 2), dtype="uint64")
#         lpossible_base_mutations = cuda.local.array((4, 2), dtype="uint64")
#         aux_kmer = cuda.local.array(KMER_LEN, dtype="uint8")
#         km = cuda.local.array(KMER_LEN, dtype="uint8")
#         aux_km2 = cuda.local.array(KMER_LEN, dtype="uint8")
#         rep = cuda.local.array(KMER_LEN, dtype="uint8")
#         encoded_bases = cuda.local.array(MAX_LEN, dtype="uint8")
#         two_sided_corrections = cuda.local.array((4, 2), dtype='int16')
#         seqlen = end - start
#         size = seqlen - kmer_len
#
#         # this should be a terminal argument
#         max_iters = 2
#         corrections_counter = 0
#
#         # we try to transfer the reads assigned for this thread into its private memory for memory access issues
#         for idx in range(0, seqlen):
#             local_reads[idx] = reads[idx + start]
#
#         for i in range(0, 4):
#             bases[i] = i + 1
#
#         for _ in range(0, max_iters):
#             num_corrections = correct_two_sided(
#                 seqlen,
#                 kmer_spectrum,
#                 km,
#                 aux_kmer,
#                 kmer_len,
#                 bases,
#                 solids,
#                 local_reads,
#                 lpossible_base_mutations,
#                 rpossible_base_mutations,
#                 size,
#                 rep,
#                 aux_km2,
#                 encoded_bases,
#
#             )
#             # this read is error free. Stop correction
#             # on first iteration, there might be correction reflected in local read, then on the second iteration all bases are solids, then before
#             # returning, we have to copy it back into global memory stored reads
#             if num_corrections == 0:
#                 copy_kmer(dev_solids[threadIdx], solids, 0, seqlen)
#                 for idx in range(0, seqlen):
#                     reads[idx + start] = local_reads[idx]
#
#                 corrections_flag[threadIdx] = 0
#                 return
#
#             # this read has more one than error within a kmer. Pass the read to one sided correction
#             if num_corrections == -1:
#                 corrections_counter = -1
#                 break
#
#             corrected_tracker[threadIdx] = 1
#             corrections_counter = 1
#         # endfor max_iters
#         copy_kmer(dev_solids[threadIdx], solids, 0, seqlen)
#         corrections_flag[threadIdx] = corrections_counter
#
#         # bring local read back to global memory reads
#
#         for idx in range(0, seqlen):
#             reads[idx + start] = local_reads[idx]
#         cuda.syncthreads()
#         return
#
@cuda.jit(device=True)
def correct_two_sided(
    seqlen,
    kmer_spectrum,
    km,
    aux_kmer,
    kmer_len,
    bases,
    solids,
    local_read,
    lpossible_base_mutations,
    rpossible_base_mutations,
    size,
    rep,
    aux_km2,
    encoded_bases,
):
    # clear solids
    for i in range(0, seqlen):
        solids[i] = -1

    # NOTE:: an encoded version of the current read
    copy_kmer(encoded_bases, local_read, 0, seqlen)
    encode_bases(encoded_bases, seqlen)

    identify_solid_bases(
        encoded_bases, kmer_len, kmer_spectrum, solids, km, size, aux_kmer
    )
    # check whether solids array does not contain -1, return 0 if yes
    if all_solid_base(solids, seqlen):
        return 0

    # for bases index 0 until size
    klen_idx = kmer_len - 1
    for ipos in range(0, size + 1):

        # trusted base
        if solids[ipos] == 1:
            continue
        lpos = 0
        # right kmer
        copy_kmer(km, encoded_bases, ipos, ipos + kmer_len)

        # left kmer
        if ipos >= kmer_len:
            copy_kmer(aux_kmer, encoded_bases, ipos - klen_idx, ipos + 1)
            lpos = klen_idx

        else:
            copy_kmer(aux_kmer, encoded_bases, 0, kmer_len)
            lpos = ipos

        # checks for reverse complement and checking lowest canonical representation
        copy_kmer(rep, km, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, km, kmer_len):
            copy_kmer(rep, km, 0, kmer_len)
            rev_comp = False

        # select all possible mutation right kmer
        rnum_bases = select_mutations(
            kmer_spectrum,
            bases,
            rep,
            kmer_len,
            0,
            rpossible_base_mutations,
            rev_comp,
            km,
            aux_km2,
        )
        # checks for reverse complement and checking lowest canonical representation
        copy_kmer(rep, aux_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, aux_kmer, kmer_len):
            copy_kmer(rep, aux_kmer, 0, kmer_len)
            rev_comp = False

        # select all possible mutations for left kmer
        lnum_bases = select_mutations(
            kmer_spectrum,
            bases,
            rep,
            kmer_len,
            lpos,
            lpossible_base_mutations,
            rev_comp,
            aux_kmer,
            aux_km2,
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

        # left kmer
        copy_kmer(aux_kmer, encoded_bases, ipos - klen_idx, ipos + 1)

        # right kmer
        copy_kmer(km, encoded_bases, size, seqlen)

        if solids[ipos] == 1:
            continue

        copy_kmer(rep, km, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, km, kmer_len):
            copy_kmer(rep, km, 0, kmer_len)
            rev_comp = False

        # select mutations for right kmer
        rnum_bases = select_mutations(
            kmer_spectrum,
            bases,
            rep,
            kmer_len,
            ipos - size,
            rpossible_base_mutations,
            rev_comp,
            km,
            aux_km2,
        )

        copy_kmer(rep, aux_kmer, 0, kmer_len)
        reverse_comp(rep, kmer_len)
        rev_comp = True

        if lower(rep, aux_kmer, kmer_len):
            copy_kmer(rep, aux_kmer, 0, kmer_len)
            rev_comp = False
        # select mutations for left kmer
        lnum_bases = select_mutations(
            kmer_spectrum,
            bases,
            rep,
            kmer_len,
            klen_idx,
            lpossible_base_mutations,
            rev_comp,
            aux_kmer,
            aux_km2,
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


# TODO::utilize all of the GPU
@cuda.jit()
def one_sided_kernel(
    kmer_spectrum,
    reads,
    offsets,
    kmer_len,
    dev_reads_result,
   
):
    threadIdx = cuda.grid(1)
    if threadIdx < offsets.shape[0]:

        # return early if two sided deems the read as solid

        MAX_LEN = 400
        DEFAULT_KMER_LEN = 18
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        # start, end = initial_start - last_end_idx, initial_end - last_end_idx
        solids = cuda.local.array(MAX_LEN, dtype="int8")
        region_indices = cuda.local.array((20, 3), dtype="int32")
        voting_matrix = cuda.local.array((MAX_LEN, 4), dtype="uint32")
        selected_bases = cuda.local.array((6, 2), dtype="uint64")
        # lpossible_base_mutations = cuda.local.array((6, 2), dtype="uint64")
        # rpossible_base_mutations = cuda.local.array((6, 2), dtype="uint64")
        km = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        aux_km = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        aux_km2 = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        rep = cuda.local.array(DEFAULT_KMER_LEN, dtype="uint8")
        bases = cuda.local.array(4, dtype="uint8")
        local_read = cuda.local.array(MAX_LEN, dtype="uint8")
        local_read_aux = cuda.local.array(MAX_LEN, dtype="uint8")
        encoded_bases = cuda.local.array(MAX_LEN, dtype="uint8")
        key = cuda.local.array(3, dtype='uint32')
        corrections = cuda.local.array((MAX_LEN, 2), dtype='int64')
        maxIters = 4
        min_vote = 3
        seqlen = end - start
        size = seqlen - kmer_len

        # seeding bases 1 to 4
        for i in range(0, 4):
            bases[i] = i + 1

        # transfer global memory store reads to local thread memory read
        for idx in range(0, seqlen):
            local_read[idx] = reads[threadIdx][idx]

        # cuda.syncthreads()
        #NOTE:: CHECKING THE EFFECT OF TWO SIDED INTO THE CORRECTION
        # for _ in range(0, 2):
        #
        #     num_corrections = correct_two_sided(
        #         seqlen,
        #         kmer_spectrum,
        #         km,
        #         aux_km,
        #         kmer_len,
        #         bases,
        #         solids,
        #         local_read,
        #         lpossible_base_mutations,
        #         rpossible_base_mutations,
        #         size,
        #         rep,
        #         aux_km2,
        #         encoded_bases,
        #     )
        #     # this read is error free. Stop correction
        #     # on first iteration, there might be correction reflected in local read, then on the second iteration all bases are solids, then before
        #     # returning, we have to copy it back into global memory stored reads
        #     if num_corrections == 0:
        #         for idx in range(0, seqlen):
        #             dev_solids[threadIdx][idx] = solids[idx]
        #             dev_reads_result[threadIdx][idx] = local_read[idx]
        #         return
        #
        #     # this read has more one than error within a kmer. Pass the read to one sided correction
        #     elif num_corrections < 0:
        #         break
        # endfor two_sided

        for nerr in range(1, maxIters + 1):
            # TODO:: to be fixed
            distance = (maxIters - nerr) + 1
            ping = False
            for _ in range(2):
                ping = not ping
                # reset solids and aux_corrections every before run of onesided
                corrections_made = onesidedv2(
                    local_read,
                    km,
                    aux_km,
                    region_indices,
                    selected_bases,
                    kmer_len,
                    seqlen,
                    size,
                    kmer_spectrum,
                    solids,
                    bases,
                    nerr,
                    distance,
                    local_read_aux,
                    rep,
                    aux_km2,
                    encoded_bases,
                    key,
                    ping,
                    corrections
                )

                # returns -1 means all bases are solid
                if corrections_made == -1:
                    for idx in range(0, seqlen):
                        dev_reads_result[threadIdx][idx] = local_read[idx]
                    return

                # no corrections made
                if corrections_made == 0:
                    break

                # a correction has been recorded and put the corrections in local read
                elif corrections_made > 0:
                    for i in range(corrections_made):
                        if corrections[i][0] > 0 and corrections[i][0] < 5 and corrections[i][1] < seqlen:
                            local_read[corrections[i][1]] = corrections[i][0]

            # start voting refinement here

            copy_kmer(encoded_bases, local_read, 0, seqlen)
            encode_bases(encoded_bases, seqlen)
            max_vote = cast_votes(
                encoded_bases,
                voting_matrix,
                seqlen,
                kmer_len,
                bases,
                seqlen - kmer_len,
                kmer_spectrum,
                km,
                rep,
                aux_km,
                aux_km2,
            )


            # the read is error free at this point
            if max_vote == 0:
                break
            elif max_vote >= min_vote:
                apply_voting_result(local_read, voting_matrix, seqlen, bases, max_vote)

        for idx in range(0, seqlen):
            dev_reads_result[threadIdx][idx] = local_read[idx]
        cuda.syncthreads()

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
def onesidedv2(
    original_read,
    ascii_kmer,
    aux_kmer,
    region_indices,
    selected_bases,
    kmer_len,
    seq_len,
    size,
    spectrum,
    solids,
    bases,
    max_corrections,
    distance,
    local_read_aux,
    rep,
    aux_km2,
    encoded_bases,
    key,
    ping,
    corrections,
):

    for idx in range(seq_len):
        solids[idx] = -1

    copy_kmer(encoded_bases, original_read, 0, seq_len)
    encode_bases(encoded_bases, seq_len)

    corrections_made = 0

    regions_count = identify_trusted_regions_v2(
        encoded_bases,
        ascii_kmer,
        rep,
        kmer_len,
        spectrum,
        seq_len,
        size,
        solids,
        region_indices,
    )
    if all_solid_base(solids, seq_len):
        return -1

    if ping:
        sort_ping(region_indices, key, regions_count)
    else:
        sort_pong(region_indices, key, regions_count)

    for region in range(0, regions_count):
        right_mer_idx = region_indices[region][1]
        right_orientation_idx = -1
        last_position = -1
        num_corrections = 0
        copy_kmer(local_read_aux, encoded_bases, 0, seq_len)
        for spos in range(right_mer_idx + 1, size + 1):
            target_pos = spos + (kmer_len - 1)
            if solids[target_pos] == 1:
                break
            copy_kmer(ascii_kmer, local_read_aux, spos, target_pos + 1)

            # checks for reverse complement and checking lowest canonical representation
            copy_kmer(rep, ascii_kmer, 0, kmer_len)
            reverse_comp(rep, kmer_len)
            rev_comp = True

            if lower(rep, ascii_kmer, kmer_len):
                copy_kmer(rep, ascii_kmer, 0, kmer_len)
                rev_comp = False

            # select all possible mutations
            kmer = transform_to_key(rep, kmer_len)
            if not in_spectrum(spectrum, kmer):
                done = False
                num_bases = select_mutations(
                    spectrum,
                    bases,
                    rep,
                    kmer_len,
                    kmer_len - 1,
                    selected_bases,
                    rev_comp,
                    aux_kmer,
                    aux_km2,
                )

                # directly apply correction if mutation is equal to 1
                if num_bases == 1:
                    if right_orientation_idx == -1:
                        right_orientation_idx = target_pos
                    corrections[corrections_made][0] = selected_bases[0][0]
                    corrections[corrections_made][1] = target_pos
                    local_read_aux[target_pos] = selected_bases[0][0]
                    corrections_made += 1
                    done = True
                else:
                    best_base = -1
                    best_base_occurence = -1
                    for idx in range(0, num_bases):
                        copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                        if successor_v2(
                            kmer_len,
                            local_read_aux,
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

                        corrections[corrections_made][0] = best_base
                        corrections[corrections_made][1] = target_pos
                        local_read_aux[target_pos] = best_base
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
                            for pos in range(num_corrections):
                                corrections[corrections_made - 1][0] = 0
                                corrections[corrections_made - 1][1] = 0
                                corrections_made -= 1

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

            # copy_kmer(local_read_aux, encoded_bases, 0, seq_len)
            last_position = right_orientation_idx

            num_corrections = 0
            pos = lkmer_idx - 1
            while pos >= 0:
                # the current base is trusted
                if solids[pos] == 1:
                    break

                copy_kmer(ascii_kmer, local_read_aux, pos, pos + kmer_len)

                copy_kmer(rep, ascii_kmer, 0, kmer_len)
                reverse_comp(rep, kmer_len)
                rev_comp = True

                if lower(rep, ascii_kmer, kmer_len):
                    copy_kmer(rep, ascii_kmer, 0, kmer_len)
                    rev_comp = False

                kmer = transform_to_key(rep, kmer_len)
                if not in_spectrum(spectrum, kmer):
                    num_bases = select_mutations(
                        spectrum,
                        bases,
                        rep,
                        kmer_len,
                        0,
                        selected_bases,
                        rev_comp,
                        aux_kmer,
                        aux_km2,
                    )
                    done = False

                    # apply correction
                    if num_bases == 1:
                        corrections[corrections_made][0] = selected_bases[0][0]
                        corrections[corrections_made][1] = pos
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
                                if selected_bases[idx][1] > best_base_occurence:
                                    best_base_occurence = selected_bases[idx][1]
                                    best_base = selected_bases[idx][0]

                        if best_base > 0 and best_base_occurence > 0:
                            corrections[corrections_made][0] = best_base
                            corrections[corrections_made][1] = pos
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
                                for _ in range(num_corrections):
                                    corrections[corrections_made - 1][0] = 0
                                    corrections[corrections_made - 1][1] = 0
                                    corrections_made -= 1

                                break
                        else:
                            last_position = pos
                            num_corrections = 0
                        pos -= 1
                        continue

                    # the correction for the current base is done == False
                    break
                # endif not in_spectrum
                pos -= 1
        # endfor lkmer_idx to 0

    # endfor regions_countsahre
    return corrections_made
