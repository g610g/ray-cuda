from test_modules import (
    in_spectrum,
    transform_to_key,
    give_kmer_multiplicity,
    mark_kmer_counter,
    lookahead_validation,
)


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
    if posibility > 1:
        pass
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    # return counter


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

    # this is already unit tested that the indexing is correct and I assume that it wont access elements out of bounds since the while loop caller of this function will stop
    # if the region_end has found neighbor region or is at the end of the index
    forward_kmer = local_reads[(region_end - (kmer_len - 1)) + 1 : region_end + 2]

    for alternative_base in bases:
        forward_kmer[-1] = alternative_base
        candidate_kmer = transform_to_key(forward_kmer, kmer_len)

        if in_spectrum(kmer_spectrum, candidate_kmer):
            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )
            possibility += 1
            alternative = alternative_base

    # Correction for this specific orientation and index is stopped
    if possibility == 0:
        print(
            f"Has {possibility} possiblity. Ending correction in index {region_end + 1}"
        )
        return False

    print(
        f"{possibility} possiblity."
    )
    # not sure if correct indexing for reads
    if possibility == 1:

        local_reads[region_end + 1] = alternative

        # increase the number correction for this kmer
        mark_kmer_counter(
            region_end + 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        max = 0
        chosen_alternative_base = -1
        chosen_alternative_base_occurence = -1
        for idx in range(possibility):
            is_potential_correction = lookahead_validation(
                kmer_len,
                local_reads,
                kmer_spectrum,
                region_end + 1,
                alternatives[idx][0],
                neighbors_max_count=2,
            )
            if is_potential_correction:
                # find greatest occurence out of all
                if alternatives[idx][1] > chosen_alternative_base_occurence:
                    chosen_alternative_base = alternatives[idx][0]
                    chosen_alternative_base_occurence = alternatives[idx][1]

        if chosen_alternative_base != -1 and chosen_alternative_base_occurence != -1:
            local_reads[region_end + 1] = alternatives[max][0]
            # increase the number correction for this kmer
            mark_kmer_counter(
                region_end + 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
            )
            return True
        

def correct_read_one_sided_left(
    local_reads,
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

    backward_kmer = local_reads[region_start - 1 : region_start + (kmer_len - 1)]

    # find alternative  base
    for alternative_base in bases:
        backward_kmer[0] = alternative_base
        candidate_kmer = transform_to_key(backward_kmer, kmer_len)

        # if the candidate kmer is in the spectrum and it passes the lookahead validation step, then the alternative base is reserved as potential correction base
        if in_spectrum(kmer_spectrum, candidate_kmer):

            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )

            possibility += 1
            alternative = alternative_base

    # returning false should cause the caller to break the loop since it fails to correct (base on the Musket paper)
    print(f"Possibility: {possibility}")
    if possibility == 0:
        return False

    # not sure if correct indexing for reads
    if possibility == 1:

        local_reads[region_start - 1] = alternative
        mark_kmer_counter(
            region_start - 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        max = 0
        chosen_alternative_base = -1
        chosen_alternative_base_occurence = -1
        for idx in range(possibility):
            is_potential_correction = lookahead_validation(
                kmer_len,
                local_reads,
                kmer_spectrum,
                region_start - 1,
                alternatives[idx][0],
                neighbors_max_count=2,
            )
            if is_potential_correction:
                # find greatest occurence out of all
                if alternatives[idx][1] > chosen_alternative_base_occurence:
                    chosen_alternative_base = alternatives[idx][0]
                    chosen_alternative_base_occurence = alternatives[idx][1]

        if chosen_alternative_base != -1 and chosen_alternative_base_occurence != -1:
            local_reads[region_start - 1] = alternatives[max][0]

            # increase the number correction for this kmer
            mark_kmer_counter(
                region_start - 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
            )
            return True
