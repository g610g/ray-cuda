from test_modules import in_spectrum, transform_to_key, give_kmer_multiplicity, mark_kmer_counter,lookahead_validation


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
    curr_kmer = local_reads[(region_end - (kmer_len - 1)) : region_end + 1]
    forward_kmer = local_reads[(region_end - (kmer_len - 1)) + 1 : region_end + 2]

    curr_kmer_transformed = transform_to_key(curr_kmer, kmer_len)
    forward_kmer_transformed = transform_to_key(forward_kmer, kmer_len)

    # false implies failure
    if in_spectrum(kmer_spectrum, curr_kmer_transformed) and not in_spectrum(
        kmer_spectrum, forward_kmer_transformed
    ):

        # find alternative  base
        for alternative_base in bases:
            forward_kmer[-1] = alternative_base
            candidate_kmer = transform_to_key(forward_kmer, kmer_len)

            #if the candidate kmer is in the spectrum and it passes the lookahead validation step, then the alternative base is reserved as potential correction base
            if in_spectrum(kmer_spectrum, candidate_kmer) and lookahead_validation(kmer_len, local_reads, kmer_spectrum, region_end + 1, alternative_base, neighbors_max_count=2):

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
        max = 0
        for idx in range(possibility):
            if alternatives[idx][1] + 1 >= alternatives[max][1] + 1:
                max = idx

        local_reads[region_end + 1] = alternatives[max][0]
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

    curr_kmer = local_reads[region_start : region_start + kmer_len]
    backward_kmer = local_reads[region_start - 1 : region_start + (kmer_len - 1)]

    curr_kmer_transformed = transform_to_key(curr_kmer, kmer_len)
    backward_kmer_transformed = transform_to_key(backward_kmer, kmer_len)

    if in_spectrum(kmer_spectrum, curr_kmer_transformed) and not in_spectrum(
        kmer_spectrum, backward_kmer_transformed
    ):
        # find alternative  base
        for alternative_base in bases:
            backward_kmer[0] = alternative_base
            candidate_kmer = transform_to_key(backward_kmer, kmer_len)

            #if the candidate kmer is in the spectrum and it passes the lookahead validation step, then the alternative base is reserved as potential correction base
            if in_spectrum(kmer_spectrum, candidate_kmer) and lookahead_validation(kmer_len, local_reads, kmer_spectrum, region_start - 1, alternative_base, neighbors_max_count=2):

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

        local_reads[region_start - 1] = alternative
        mark_kmer_counter(
            region_start - 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:

        max = 0
        for idx in range(possibility):
            if alternatives[idx][1] + 1 >= alternatives[max][1] + 1:
                max = idx

        local_reads[region_start - 1] = alternatives[max][0]
        mark_kmer_counter(
            region_start - 1, kmer_tracker, kmer_len, max_kmer_idx, read_length
        )
        return True
