def in_spectrum(spectrum, target_kmer):
    for kmer in spectrum:
        if target_kmer == kmer[0]:
            return True
    return False


def mark_solids_array(solids, start, end):

    for i in range(start, end):
        solids[i] = 1


def transform_to_key(ascii_kmer, len):
    multiplier = 1
    key = 0
    while len != 0:
        key += ascii_kmer[len - 1] * multiplier
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
    if not in_spectrum(kmer_spectrum, kmer):
        return -1

    return 100


# The successor implementation of lookahead validation
def lookahead_successor(
    kmer_length,
    local_read,
    kmer_spectrum,
    modified_base_idx,
    alternative_base,
    neighbors_max_count=2,
):

    available_neighbors = min(
        modified_base_idx, kmer_length, len(local_read) - modified_base_idx
    )

    # if no neighbors
    if (
        modified_base_idx > len(local_read) - kmer_length
        or modified_base_idx < kmer_length
    ):
        print("Something error went wrong")
        return True
    # start index is after forward kmer. (That's why -2)
    # I should calculate the number of neighbors available to check and the stride of neighbors to check
    start_idx = modified_base_idx - (kmer_length - 2)

    # calculated number of neighbors
    max_end_idx = modified_base_idx
    counter = kmer_length - 2
    neighbors_traversed = 0

    for idx in range(start_idx, max_end_idx + 1):

        # I might try to limit the max neighbors to be traversed
        if neighbors_traversed >= neighbors_max_count:
            break

        alternative_kmer = local_read[idx : idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            print(f"{transformed_alternative_kmer} is not in spectrum")
            return False
        print(
            f"counter: {counter}, start idx: {start_idx}, max end idx: {max_end_idx}, neighbors available: {available_neighbors}"
        )
        print(f"{transformed_alternative_kmer} exists")
        counter -= 1
        neighbors_traversed += 1

    return True


def predeccessor_revised(
    kmer_length,
    local_read,
    kmer_spectrum,
    seq_len,
    alternative_base,
    neighbors_count,
):

    if seq_len <= 0 or seq_len > (len(local_read) - kmer_length):
        print("Something error went wrong")
        return True

    counter = 1
    neighbors_traversed = 0

    for idx in range(seq_len, -1, -1):
        if neighbors_traversed >= neighbors_count:
            break

        alternative_kmer = local_read[idx : idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False

        neighbors_traversed += 1
        counter += 1
    return True

def predeccessor(
    kmer_length, local_read, kmer_spectrum, target_pos, alternative_base, max_traverse
):
    ipos = target_pos - 1
    if ipos <= 0:
        return True
    counter = 1
    traversed_count = 0
    for idx in range(ipos, -1, -1):
        if traversed_count >= max_traverse:
            return True

        alternative_kmer = local_read[idx : idx + kmer_length]
        print(f"start: {idx} end: {idx + kmer_length}")
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)

        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False

        counter += 1
        traversed_count += 1

    return True

def lookahead_predeccessor(
    kmer_length,
    local_read,
    kmer_spectrum,
    modified_base_idx,
    alternative_base,
    neighbors_count,
):
    # available_neighbors = min(
    #     kmer_length, modified_base_idx, len(local_read) - modified_base_idx
    # )
    neighbors_traversed = 0

    # starting index for modified base within preceeding kmer
    counter = 1

    # when modified base idx is zero, it means no available neighbors
    if modified_base_idx <= 0:
        return True

    idx = modified_base_idx - 1
    print(f"Within predeccesor with alternative base:{alternative_base}")
    while idx >= 0:
        if neighbors_traversed >= neighbors_count:
            break
        alternative_kmer = local_read[idx : idx + kmer_length]
        print(f"Kmer starting from index: {idx} to index: {idx + kmer_length}")
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False

        idx -= 1
        counter += 1
        neighbors_traversed += 1

    return True


def successor(
    kmer_length, local_read, kmer_spectrum, target_pos, alternative_base, max_traverse
):
    # edge cases
    if target_pos > len(local_read) - kmer_length:
        return True

    # ipos represents the starting index for successors kmers
    ipos = target_pos - (kmer_length - 2)
    traversed_count = 0
    counter = kmer_length - 2
    for idx in range(ipos, target_pos):
        if traversed_count >= max_traverse:
            return True

        alternative_kmer = local_read[idx : idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)

        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False
        counter -= 1
        traversed_count += 1

    return True

def generate_kmers(read, kmer_length, kmer_spectrum):
    for idx in range(0, len(read) - (kmer_length - 1)):
        kmer_spectrum.append(
            transform_to_key(read[idx : idx + kmer_length], kmer_length)
        )


def count_occurence(spectrum):
    new_spectrum = []
    for kmer in spectrum:
        occurence = 0
        for inner_kmer in spectrum:
            if kmer == inner_kmer:
                occurence += 1
        new_spectrum.append([kmer, occurence])

    return new_spectrum.copy()
