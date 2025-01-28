from test_modules import (
    in_spectrum,
    transform_to_key,
    give_kmer_multiplicity,
    mark_kmer_counter,
    successor,
    predeccessor

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
    local_read,
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

    target_pos = region_end + 1
    ipos = target_pos - (kmer_len - 1)

    forward_kmer = local_read[ipos: target_pos]

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

        local_read[region_end + 1] = alternative
        mark_kmer_counter(
            target_pos, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
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
                target_pos,
                alternatives[idx][0],
                2,
            )
            print(f"is potential correction (orientation right) -> {target_pos}: {is_potential_correction}")
            if is_potential_correction:
                if alternatives[idx][1] > choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            mark_kmer_counter(
                region_end + 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
            )
            return True
        return False


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
    print(f"Possibility: {possibility}")
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
            is_potential_correction = predeccessor(
                kmer_len,
                local_read,
                kmer_spectrum,
                target_pos,
                alternatives[idx][0],
                2,
            )
            print(f"is potential correction (orientation left) -> {target_pos}: {is_potential_correction} alternative base: {alternatives[idx][0]}")
            if is_potential_correction:
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

