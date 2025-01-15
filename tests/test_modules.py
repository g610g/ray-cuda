def in_spectrum(spectrum, kmer):
    if kmer in spectrum:
        return True
    return False

def mark_solids_array(solids, start, end):

    for i in range(start, end):
        solids[i] = 1

def transform_to_key(ascii_kmer, len):
    multiplier = 1
    key = 0
    while(len != 0):
        key += (ascii_kmer[len - 1] * multiplier)
        multiplier *= 10
        len -= 1

    return key
def identify_solid_bases(local_reads, start, end, kmer_len, kmer_spectrum, solids):

    for idx in range(0, (end - start) - (kmer_len - 1)):
        ascii_kmer = local_reads[idx : idx + kmer_len]

        curr_kmer = transform_to_key(ascii_kmer, kmer_len)

        # set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx, (idx + kmer_len))

def identify_trusted_regions(
    start, end, kmer_spectrum, local_reads, kmer_len, region_indices, solids
):

    identify_solid_bases(local_reads, start, end, kmer_len, kmer_spectrum, solids)

    current_indices_idx = 0
    base_count = 0
    region_start = 0
    region_end = 0

    # idx will be a relative index
    for idx in range(end - start):

        # a trusted region has been found. Append it into the identified regions
        if base_count >= kmer_len and solids[idx] == -1:

            (
                region_indices[current_indices_idx][0],
                region_indices[current_indices_idx][1],
            ) = (region_start, region_end)
            region_start = idx + 1
            region_end = idx + 1
            current_indices_idx += 1
            base_count = 0

        # reset the region start since its left part is not a trusted region anymore
        if solids[idx] == -1 and base_count < kmer_len:
            region_start = idx + 1
            region_end = idx + 1
            base_count = 0

        if solids[idx] == 1:
            region_end = idx
            base_count += 1

    # ending
    if base_count >= kmer_len:

        (
            region_indices[current_indices_idx][0],
            region_indices[current_indices_idx][1],
        ) = (region_start, region_end)
        current_indices_idx += 1

    # this will be the length or the number of trusted regions
    return current_indices_idx





def mark_kmer_counter(base_idx, kmer_counter_list, kmer_len, max_kmer_idx, read_length):
    if base_idx < kmer_len:
        for idx in range(0, base_idx + 1):
            kmer_counter_list[idx] += 1
        return

    if base_idx > (read_length - (kmer_len - 1)):
        min = base_idx - (kmer_len - 1)
        for idx in range(min, max_kmer_idx + 1):
            kmer_counter_list[idx] += 1
        return

    min = base_idx - (kmer_len - 1)
    if base_idx > max_kmer_idx:
        for idx in range(min, max_kmer_idx + 1):
            kmer_counter_list[idx] += 1
        return
    for idx in range(min, base_idx + 1):
        kmer_counter_list[idx] += 1
    return

def give_kmer_multiplicity(kmer_spectrum, kmer):
    return 100

def lookahead_validation(
    kmer_length,
    local_read,
    kmer_spectrum,
    modified_base_idx,
    alternative_base,
    neighbors_max_count=2,
):
    # this is for base that has kmers that covers < neighbors_max_count
    if modified_base_idx < neighbors_max_count:
        num_possible_neighbors = modified_base_idx + 1
        counter = modified_base_idx
        min_idx = 0
        for _ in range(num_possible_neighbors):
            alternative_kmer = local_read[min_idx: min_idx+kmer_length]
            alternative_kmer[counter] = alternative_base

            transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
            if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
                return False
            min_idx += 1
            counter -= 1
    # for bases that are modified outside the "easy range"
    if modified_base_idx >= len(local_read) - kmer_length:
        num_possible_neighbors = (len(local_read) - 1) - modified_base_idx

        min_idx = modified_base_idx - (kmer_length - 1)
        max_idx = modified_base_idx
        counter = kmer_length - 1

        for _ in range(num_possible_neighbors):
            alternative_kmer = local_read[min_idx: min_idx + kmer_length]
            alternative_kmer[counter] = alternative_base
            transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
            if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
                return False
            min_idx += 1
            counter -= 1

    if modified_base_idx < (kmer_length - 1):
        min_idx = 0
        max_idx = kmer_length
        counter = modified_base_idx
    else:
        # this is the modified base idx that are within the range of "easy range"
        min_idx = modified_base_idx - (kmer_length - 1)
        max_idx = modified_base_idx
        counter = kmer_length - 1

    print(f"counter: {counter} min idx: {min_idx} max idx: {max_idx}")
    for _idx in range(neighbors_max_count):
        if min_idx > max_idx:
            return False
        alternative_kmer = local_read[min_idx : min_idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False

        min_idx += 1
        counter -= 1

    # returned True meaning the alternative base which sequencing error occurs is (valid)?
    return True
