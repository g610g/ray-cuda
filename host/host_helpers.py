def all_solid_base(solids, length):
    for idx in range(length):
       if solids[idx] == -1:
           return False

def give_kmer_multiplicity(kmer_spectrum, kmer):

    index = binary_search_2d(kmer_spectrum, kmer)
    if index != -1:
        return kmer_spectrum[index][1]

    return index
def select_mutations(spectrum, bases, ascii_kmer, kmer_len, pos, selected_bases):
    num_bases = 0 
    original_base = ascii_kmer[pos]
    for idx in range(4):
        if bases[idx] == original_base:
            continue
        else:
            ascii_kmer[pos] = bases[idx]
            candidate = transform_to_key(ascii_kmer, kmer_len)
            if in_spectrum(spectrum, candidate):
                selected_bases[num_bases] = bases[idx]
                num_bases += 1

    #assign back the original base 
    ascii_kmer[pos] = original_base

    return num_bases
def mark_solids_array(solids, start, end):

    for i in range(start, end):
        solids[i] = 1

def binary_search_2d(sorted_arr, needle):
    sorted_arr_len = len(sorted_arr)
    right = sorted_arr_len - 1
    left = 0

    while left <= right:
        middle = (left + right) // 2
        if sorted_arr[middle][0] == needle:
            return middle

        elif sorted_arr[middle][0] > needle:
            right = middle - 1

        elif sorted_arr[middle][0] < needle:
            left = middle + 1
    return -1
def in_spectrum(spectrum, kmer):
    if binary_search_2d(spectrum, kmer) != -1:
        return True

    return False
def transform_to_key(ascii_kmer):
    return int("".join(map(str, ascii_kmer)))

def copy_kmer(aux_kmer, local_read, start, end):
    for i in range(start, end):
        aux_kmer[i - start] = local_read[i]
def identify_solid_bases(local_reads, kmer_len, kmer_spectrum, solids, ascii_kmer, size):

    for idx in range(0, size + 1):
        ascii_kmer = local_reads[idx: idx + kmer_len].copy()
        curr_kmer = transform_to_key(ascii_kmer)

        # set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx, idx + kmer_len)

def identify_trusted_regions(
    seq_len, kmer_spectrum, local_reads, kmer_len, region_indices, solids, aux_kmer, size
):

    identify_solid_bases(local_reads, kmer_len, kmer_spectrum, solids, aux_kmer, size)

    current_indices_idx = 0
    base_count = 0
    region_start = 0
    region_end = 0

    # idx will be a relative index
    for idx in range(seq_len):

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


def successor_v2(
    kmer_length, local_read, aux_kmer, kmer_spectrum , alternative_base, spos, distance, read_len
):
    seqlen = read_len - spos - 1
    offset = spos + 1

    # edge cases
    if seqlen < kmer_length or distance <= 0:
        return True

    end_idx = min(seqlen - kmer_length, distance - 1)
    idx = 0
    counter = kmer_length - 2
    while idx <= end_idx:
        copy_kmer(aux_kmer, local_read, offset + idx, offset + idx + kmer_length - 1)
        aux_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(aux_kmer, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False
        counter -= 1
        idx += 1

    return True
def predeccessor_v2(
    kmer_length, local_read, aux_kmer, kmer_spectrum, target_pos, alternative_base, distance
):
    ipos = target_pos - 1
    if ipos <= 0 or distance <= 0:
        return True
    spos = max(0, ipos - distance)

    counter = 2
    idx = ipos - 1
    while idx >= spos:
        if counter < kmer_length:
            copy_kmer(aux_kmer, local_read, idx, idx + kmer_length)
            aux_kmer[counter] = alternative_base
            candidate = transform_to_key(aux_kmer, kmer_length)
            if not in_spectrum(kmer_spectrum, candidate):
                return False
            counter += 1
            idx -= 1
    return True
