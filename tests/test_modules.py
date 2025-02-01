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
    length = end - start
    endpos = length - (kmer_len - 1)

    for idx in range(0, endpos):
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


def give_kmer_multiplicity(kmer_spectrum, kmer):
    if not in_spectrum(kmer_spectrum, kmer):
        return -1

    return 100


def predeccessor(
    kmer_length, local_read, kmer_spectrum, target_pos, alternative_base, max_traverse
):
    ipos = target_pos - 1
    if ipos <= 0:
        return True
    counter = 1
    traversed_count = 0
    print(f"running predeccessor for indices:")
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
def predeccessor_v2(
    kmer_length, local_read, kmer_spectrum, target_pos, alternative_base, distance
):
    print(f"predeccesor distance: {distance}")
    ipos = target_pos - 1
    if ipos <= 0 or distance <= 0:
        return True
    spos = max(0, ipos - distance)

    counter = kmer_length - 2
    for idx in range(ipos - 1, spos - 1, -1):
        if counter < kmer_length:
            ascii_kmer = local_read[idx: idx + kmer_length]
            ascii_kmer[counter] = alternative_base
            candidate = transform_to_key(ascii_kmer, kmer_length)
            if not in_spectrum(candidate, kmer_spectrum):
                return False
            counter += 1
        else:
            ascii_kmer = local_read[idx: idx + kmer_length]
            candidate = transform_to_key(ascii_kmer, kmer_length)
            if not in_spectrum(candidate, kmer_spectrum):
                return False



def successor(
    kmer_length, local_read, kmer_spectrum , alternative_base, max_traverse, ipos
):
    seqlen = len(local_read) - ipos - 1
    offset = ipos + 1
    # edge cases
    if seqlen < kmer_length:
        return True
    end_idx = seqlen - kmer_length
    idx = 0
    traversed_count = 0
    counter = kmer_length - 2
    print("running successor in index:")
    while idx <= end_idx:
        if traversed_count >= max_traverse:
            return True

        print(f"Start: {offset + idx} end: {offset + idx + kmer_length - 1}")
        alternative_kmer = local_read[offset + idx : offset + idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)

        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False
        counter -= 1
        traversed_count += 1
        idx += 1

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

def check_solids_cardinality(solids, length):
    for idx in range(length):
       if solids[idx] == -1:
           return False
    return True
