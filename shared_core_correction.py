from numba import cuda
from shared_helpers import (
    check_tracker,
    identify_solid_bases,
    identify_trusted_regions,
    lookahead_successor,
    lookahead_predeccessor,
    mark_kmer_counter,
    predeccessor_revised,
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
        max_corrections = 2
        # we try to transfer the reads assigned for this thread into its private memory for memory access issues
        for idx in range(0, end - start):
            local_reads[idx] = reads[idx + start]
        for i in range(4):
            bases[i] = i + 1

        for _ in range(max_corrections):
            for i in range(end - start):
                solids[i] = -1

            # identify whether base is solid or not
            identify_solid_bases(
                local_reads, start, end, kmer_len, kmer_spectrum, solids
            )

            # check whether base is potential for correction
            # kulang pani diria sa pag check sa first and last bases
            for base_idx in range(0, end - start):
                # the base needs to be corrected
                if (
                    solids[base_idx] == -1
                    and base_idx >= (kmer_len - 1)
                    and base_idx <= (end - start) - kmer_len
                ):

                    left_portion = local_reads[base_idx - (kmer_len - 1) : base_idx + 1]
                    right_portion = local_reads[base_idx : base_idx + kmer_len]

                    correct_reads_two_sided(
                        base_idx,
                        local_reads,
                        kmer_len,
                        kmer_spectrum,
                        bases,
                        left_portion,
                        right_portion,
                    )

                # the leftmost bases of the read
                if solids[base_idx] == -1 and base_idx < (kmer_len - 1):
                    pass
                # the rightmost bases of the read
                if solids[base_idx] == -1 and base_idx > (end - start) - kmer_len:
                    pass

            # copy the reads from private memory back to the global memory
            for idx in range(0, end - start):
                reads[idx + start] = local_reads[idx]


@cuda.jit(device=True)
def correct_reads_two_sided(
    idx,
    local_reads,
    kmer_len,
    kmer_spectrum,
    bases,
    left_kmer,
    right_kmer,
):
    current_base = local_reads[idx]
    posibility = 0
    candidate = -1

    for alternative_base in bases:
        if alternative_base != current_base:

            # array representation
            left_kmer[-1] = alternative_base
            right_kmer[0] = alternative_base

            # whole number representation
            candidate_left = transform_to_key(left_kmer, kmer_len)
            candidate_right = transform_to_key(right_kmer, kmer_len)

            # the alternative base makes our kmers trusted
            if in_spectrum(kmer_spectrum, candidate_left) and in_spectrum(
                kmer_spectrum, candidate_right
            ):
                posibility += 1
                candidate = alternative_base

    if posibility == 1:
        local_reads[idx] = candidate
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    if posibility == 0:
        pass
        # corrected_counter[threadIdx][counter] = 10
        # counter += 1

    # ignore the correction if more than one possibility
    # check whether base is potential for correction
    # kulang pani diria sa pag check sa first and last bases
    if posibility > 1:
        pass
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    # return counter


# no implementation for tracking how many corrections are done for each kmers in the read
@cuda.jit()
def one_sided_kernel(
    kmer_spectrum,
    reads,
    offsets,
    kmer_len,
    not_corrected_counter,
    kmers_tracker,
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
        max_correction = 4

        bases = cuda.local.array(5, dtype="uint8")

        # seeding bases 1 to 4
        for i in range(4):
            bases[i] = i + 1

        # transfer global memory store reads to local thread memory read
        for idx in range(end - start):
            local_read[idx] = reads[idx + start]
            original_read[idx] = reads[idx + start]

        for _ in range(max_correction):
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

            for region in range(regions_count):
                # 1. goes toward right orientation

                # there is no next region
                if region == (regions_count - 1):
                    region_end = region_indices[region][1]

                    # while we are not at the end base of the read
                    while region_end != (end - start) - 1:
                        if not correct_read_one_sided_right(
                            local_read,
                            region_end,
                            kmer_spectrum,
                            kmer_len,
                            bases,
                            alternatives,
                            correction_tracker,
                            num_kmers - 1,
                            end - start,
                        ):
                            not_corrected_counter[threadIdx] += 1
                            break

                        # extend the portion of region end for successful correction
                        else:
                            region_end += 1
                            region_indices[region][1] = region_end

                # there is a next region
                if region != (regions_count - 1):
                    region_end = region_indices[region][1]
                    next_region_start = region_indices[region + 1][0]

                    # the loop will not stop until it does not find another region
                    while region_end != (next_region_start - 1):
                        if not correct_read_one_sided_right(
                            local_read,
                            region_end,
                            kmer_spectrum,
                            kmer_len,
                            bases,
                            alternatives,
                            correction_tracker,
                            num_kmers - 1,
                            end - start,
                        ):
                            # fails to correct this region and on this orientation
                            not_corrected_counter[threadIdx] += 1
                            break

                        # extend the portion of region end for successful correction
                        else:
                            region_end += 1
                            region_indices[region][1] = region_end

                # going towards left of the region
                # we are the leftmost region
                if region - 1 == -1:
                    region_start = region_indices[region][0]

                    # while we are not at the first base of the read
                    while region_start != 0:
                        if not correct_read_one_sided_left(
                            local_read,
                            region_start,
                            kmer_spectrum,
                            kmer_len,
                            bases,
                            alternatives,
                            correction_tracker,
                            num_kmers - 1,
                            end - start,
                        ):
                            not_corrected_counter[threadIdx] += 1
                            break
                        else:
                            region_start -= 1
                            region_indices[region][0] = region_start

                # there is another region in the left side of this region
                if region - 1 != -1:
                    region_start, prev_region_end = (
                        region_indices[region][0],
                        region_indices[region - 1][1],
                    )
                    while region_start - 1 != (prev_region_end):

                        if not correct_read_one_sided_left(
                            local_read,
                            region_start,
                            kmer_spectrum,
                            kmer_len,
                            bases,
                            alternatives,
                            correction_tracker,
                            num_kmers - 1,
                            end - start,
                        ):
                            not_corrected_counter[threadIdx] += 1
                            break
                        else:
                            region_start -= 1
                            region_indices[region][0] = region_start

            # start of the voting based refinement(havent integrated tracking kmer during apply_vm_result function)
            # for idx in range(start, end - (kmer_len - 1)):
            #     curr_kmer = transform_to_key(local_read[idx : idx + kmer_len], kmer_len)
            #
            #     # invoke voting if the kmer is not in spectrum
            #     if not in_spectrum(kmer_spectrum, curr_kmer):
            #         invoke_voting(
            #             voting_matrix,
            #             kmer_spectrum,
            #             bases,
            #             kmer_len,
            #             idx,
            #             local_read,
            #             start,
            #         )
            # # apply the result of the vm into the reads
            # apply_vm_result(voting_matrix, local_read, start, end)
        # endfor

        # checking if there are kmers that has been corrected more than once (used for debugging)
        for idx in range(num_kmers):
            kmers_tracker[threadIdx][idx] = correction_tracker[idx]

        # reverts kmers that exceeds maximum allowable corrections (default=4)

        check_tracker(
            num_kmers,
            correction_tracker,
            max_correction,
            original_read,
            local_read,
            kmer_len,
        )

        # copies back corrected local read into global memory stored reads
        for idx in range(end - start):
            reads[idx + start] = local_read[idx]


# for orientation going to the right of the read
@cuda.jit(device=True)
def correct_read_one_sided_right(
    local_reads,
    region_end,
    kmer_spectrum,
    kmer_len,
    bases,
    alternatives,
    kmer_tracker,
    max_kmer_idx,
    read_length,
):

    possibility = 0
    alternative = -1

    forward_kmer = local_reads[(region_end - (kmer_len - 1)) + 1 : region_end + 2]

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

        local_reads[region_end + 1] = alternative
        mark_kmer_counter(
            region_end + 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            is_potential_correction = lookahead_successor(
                kmer_len,
                local_reads,
                kmer_spectrum,
                region_end + 1,
                alternatives[idx][0],
                2,
            )
            if is_potential_correction:
                if alternatives[idx][1] > choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_reads[region_end + 1] = choosen_alternative_base
            mark_kmer_counter(
                region_end + 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
            )
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
    kmer_tracker,
    max_kmer_idx,
    read_length,
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
        mark_kmer_counter(target_pos, kmer_tracker, kmer_len, max_kmer_idx, read_length)
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            # is_potential_correction = predeccessor_revised(
            #     kmer_len,
            #     local_read,
            #     kmer_spectrum,
            #     target_pos - 1,
            #     alternatives[idx][0],
            #     2,
            # )
            if True:
                if alternatives[idx][1] > choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            mark_kmer_counter(
                target_pos, kmer_tracker, kmer_len, max_kmer_idx, read_length
            )
            return True
        return False
