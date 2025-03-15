import math

from numba.parfors.parfor import lower_parfor_sequential
import cudf
from numba import cuda
from helpers import (
    give_kmer_multiplicity,
    in_spectrum,
    transform_to_key,
    mark_solids_array,
)
import ray


# function that checks the kmer tracker and reverse back kmers thats been corrected greater than max allowed
@cuda.jit(device=True)
def check_tracker(
    num_kmers,
    kmer_tracker,
    max_corrections_allowed,
    original_read,
    local_read,
    kmer_len,
):
    for idx in range(num_kmers):
        if kmer_tracker[idx] > max_corrections_allowed:
            for base_idx in range(idx, idx + kmer_len):
                local_read[base_idx] = original_read[base_idx]


@cuda.jit(device=True)
def identify_solid_bases(encoded_base, kmer_len, kmer_spectrum, solids, km, size, rep):

    for idx in range(0, size + 1):
        copy_kmer(km, encoded_base, idx, idx + kmer_len)
        copy_kmer(rep, km, 0, kmer_len)

        # transform rep kmer into reverse complement representation
        reverse_comp(rep, kmer_len)

        # consider who is lexicographically smaller
        if lower(rep, km, kmer_len):
            copy_kmer(rep, km, 0, kmer_len)

        curr_kmer = transform_to_key(rep, kmer_len)

        # use the lexicographical small representation and set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx, idx + kmer_len)


@cuda.jit(device=True)
def identify_trusted_regions_v2(
    local_read, km, rep, kmer_len, spectrum, seq_len, size, solids, regions
):
    left_kmer, right_kmer = -1, -1
    solid_region = False
    regions_count = 0

    for pos in range(seq_len):
        solids[pos] = -1

    for ipos in range(0, size + 1):
        copy_kmer(km, local_read, ipos, ipos + kmer_len)
        copy_kmer(rep, local_read, ipos, ipos + kmer_len)
        reverse_comp(rep, kmer_len)
        if lower(rep, km, kmer_len):
            copy_kmer(rep, km, 0, kmer_len)

        kmer = transform_to_key(rep, kmer_len)

        if in_spectrum(spectrum, kmer):
            if not solid_region:
                solid_region = True
                left_kmer = ipos
                right_kmer = ipos
            else:
                right_kmer += 1

            for idx in range(ipos, ipos + kmer_len):
                solids[idx] = 1
        else:
            if left_kmer >= 0:
                regions[regions_count][0] = left_kmer
                regions[regions_count][1] = right_kmer
                regions[regions_count][2] = right_kmer - left_kmer
                regions_count += 1
                left_kmer = right_kmer = -1
            solid_region = False

    if solid_region and left_kmer >= 0:
        regions[regions_count][0] = left_kmer
        regions[regions_count][1] = right_kmer
        regions[regions_count][2] = right_kmer - left_kmer
        regions_count += 1
    return regions_count


@cuda.jit(device=True)
def identify_trusted_regions(
    seq_len,
    kmer_spectrum,
    local_reads,
    kmer_len,
    region_indices,
    solids,
    aux_kmer,
    size,
    rep,
):

    identify_solid_bases(
        local_reads, kmer_len, kmer_spectrum, solids, aux_kmer, size, rep
    )

    trusted_regions_count = 0
    base_count = 0
    region_start = 0
    region_end = 0

    for idx in range(seq_len):
        if solids[idx] == 1:
            region_end = idx
            base_count += 1
        else:
            if base_count >= kmer_len:
                (
                    region_indices[trusted_regions_count][0],
                    region_indices[trusted_regions_count][1],
                ) = (region_start, region_end)
                trusted_regions_count += 1
            region_start = idx + 1
            region_end = idx + 1
            base_count = 0

    if base_count >= kmer_len:

        (
            region_indices[trusted_regions_count][0],
            region_indices[trusted_regions_count][1],
        ) = (region_start, region_end)

        trusted_regions_count += 1

    return trusted_regions_count


@cuda.jit(device=True)
def to_local_reads(reads_1d, local_reads, start, end):

    for idx in range(start, end):
        local_reads[idx - start] = reads_1d[idx]


@ray.remote(num_cpus=1, num_gpus=1)
def back_to_sequence_helper(reads, offsets):

    # find reads max length
    offsets_df = cudf.DataFrame({"start": offsets[:, 0], "end": offsets[:, 1]})
    offsets_df["length"] = offsets_df["end"] - offsets_df["start"]
    max_segment_length = offsets_df["length"].max()
    print(f"max segment length: {max_segment_length}")
    cuda.profile_start()
    start = cuda.event()
    end = cuda.event()
    start.record()

    dev_reads = cuda.to_device(reads)
    dev_offsets = cuda.to_device(offsets)
    dev_reads_result = cuda.device_array(
        (offsets.shape[0], max_segment_length), dtype="uint8"
    )
    tpb = 1024
    bpg = (offsets.shape[0] + tpb) // tpb

    back_sequence_kernel[bpg, tpb](dev_reads, dev_offsets, dev_reads_result)

    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the back to sequence kernel:  {transfer_time} ms")
    cuda.profile_stop()

    return dev_reads.copy_to_host()


@cuda.jit
def back_sequence_kernel(reads, offsets, reads_result):
    threadIdx = cuda.grid(1)
    MAX_LEN = 300
    local_reads = cuda.local.array(MAX_LEN, dtype="uint8")
    if threadIdx < offsets.shape[0]:
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        seqlen = end - start

        # to_local_reads(reads, local_reads, start, end)
        to_decimal_ascii(reads[threadIdx], seqlen)

        # copy the assigned read for this thread into the 2d reads_result
        # for idx in range(seqlen):
        #     reads_result[threadIdx][idx] = local_reads[idx]
        #


# normal python function for giving insights by differentiating before and after solids
def differ_solids(solids_before, solids_after):
    if len(solids_before) != len(solids_after):
        return False
    for solid_before, solid_after in zip(solids_before, solids_after):
        differ_count = 0
        for base_before, base_after in zip(solid_before, solid_after):
            if base_before != base_after:
                differ_count += 1
        if differ_count != 0:
            print(f"difference is {differ_count}")
    return True


def print_solids_after(solids_after):
    for solid_after in solids_after:
        print(solid_after)


def count_untrusted_bases(solids_after):
    for solid_after in solids_after:
        untrusted_bases_count = 0
        for base in solid_after:
            if base == -1:
                untrusted_bases_count += 1
        if untrusted_bases_count != 0 or untrusted_bases_count > 0:
            print(f"Number of untrusted bases: {untrusted_bases_count}")


@ray.remote(num_cpus=1)
def count_error_reads(solids_batch, len):
    error_reads = 0
    for solid in solids_batch:
        for idx in range(len):
            if solid[idx] == -1:
                error_reads += 1
                print(solid)
                break
    print(f"error reads detected: {error_reads}")


@cuda.jit(device=True)
def successor_v2(
    kmer_length,
    aux_local_read,
    aux_kmer,
    kmer_spectrum,
    alternative_base,
    spos,
    distance,
    read_len,
    rep,
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
        # forward_base(aux_kmer, local_read[offset + idx + kmer_length -1], kmer_length)
        copy_kmer(aux_kmer, aux_local_read, offset + idx, offset + idx + kmer_length)
        aux_kmer[counter] = alternative_base

        # check which is lexicographically smaller between original succeeding kmer and reverse complemented succeeding kmer
        copy_kmer(rep, aux_kmer, 0, kmer_length)
        reverse_comp(rep, kmer_length)

        # choose lexicographically smaller
        if lower(rep, aux_kmer, kmer_length):
            copy_kmer(rep, aux_kmer, 0, kmer_length)

        transformed_alternative_kmer = transform_to_key(rep, kmer_length)
        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False
        counter -= 1
        idx += 1

    return True


# lookahead validation of succeeding kmers
@cuda.jit(device=True)
def successor(
    kmer_length, local_read, kmer_spectrum, alternative_base, max_traverse, ipos
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
    while idx <= end_idx:
        if traversed_count >= max_traverse:
            return True

        alternative_kmer = local_read[offset + idx : offset + idx + kmer_length]
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)

        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False
        counter -= 1
        traversed_count += 1
        idx += 1

    return True


# problematic son of a btch
@cuda.jit(device=True)
def predeccessor_v2(
    kmer_length,
    local_read,
    aux_kmer,
    kmer_spectrum,
    target_pos,
    alternative_base,
    distance,
    rep,
):
    # ipos is the first preceeding neighbor
    # ipos = target_pos - 1
    if target_pos <= 0 or distance <= 0:
        return True

    spos = max(0, target_pos - distance)
    counter = 1
    idx = target_pos - 1
    while idx >= spos:
        copy_kmer(aux_kmer, local_read, idx, idx + kmer_length)
        aux_kmer[counter] = alternative_base
        copy_kmer(rep, aux_kmer, 0, kmer_length)
        reverse_comp(rep, kmer_length)
        if lower(rep, aux_kmer, kmer_length):
            copy_kmer(rep, aux_kmer, 0, kmer_length)

        candidate = transform_to_key(rep, kmer_length)

        if not in_spectrum(kmer_spectrum, candidate):
            return False
        counter += 1
        idx -= 1
    return True


# lookahead validation of preceeding kmers
@cuda.jit(device=True)
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
        alternative_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(alternative_kmer, kmer_length)

        if not in_spectrum(kmer_spectrum, transformed_alternative_kmer):
            return False

        counter += 1
        traversed_count += 1

    return True


@cuda.jit(device=True)
def all_solid_base(solids, seqlen):
    for idx in range(seqlen):
        if solids[idx] == -1:
            return False

    return True


# copy contents of local read from its start to end
@cuda.jit(device=True)
def copy_kmer(aux_kmer, local_read, start, end):
    for i in range(start, end):
        aux_kmer[i - start] = local_read[i]


# NOTE:: semantically and syntactically correct
@cuda.jit(device=True)
def select_mutations(
    spectrum, bases, km, kmer_len, pos, selected_bases, rev_comp, aux_km, aux_km2
):
    # calculate the pos value considering rev_comp flag
    num_bases = 0
    base_index = pos
    if rev_comp and pos < kmer_len:
        base_index = (kmer_len - 1) - pos

    original_base = km[base_index]
    copy_kmer(aux_km, km, 0, kmer_len)

    for idx in range(4):
        if bases[idx] == original_base:
            continue
        else:
            # consider the reverse complement of mutated aux_km and store in aux_km2

            aux_km[base_index] = bases[idx]
            copy_kmer(aux_km2, aux_km, 0, kmer_len)
            reverse_comp(aux_km2, kmer_len)

            # check if whose lexicographically smaller between auxKm2 and auxkm
            if lower(aux_km2, aux_km, kmer_len):
                copy_kmer(aux_km2, aux_km, 0, kmer_len)
            # 234 -> 123 134
            # 123 -> 124
            # 124 -> 134
            # = 434, selected_bases = (1, occurence)
            # use the lexicographically small during checking the spectrum
            candidate = transform_to_key(aux_km2, kmer_len)
            if in_spectrum(spectrum, candidate):
                # use complement of selected base if rev comp flag is active
                base = bases[idx]
                if rev_comp:
                    base = complement(bases[idx])

                # add range restriction
                if base > 0 and base < 5:
                    selected_bases[num_bases][0] = base
                    selected_bases[num_bases][1] = give_kmer_multiplicity(
                        spectrum, candidate
                    )
                    num_bases += 1

    return num_bases


# backward the base or shifts bases to the left
@cuda.jit(device=True)
def backward_base(ascii_kmer, base, kmer_length):
    for idx in range(kmer_length - 1, 0, -1):
        ascii_kmer[idx] = ascii_kmer[(idx - 1)]

    ascii_kmer[0] = base


# forward the base or shifts bases to the right
@cuda.jit(device=True)
def forward_base(ascii_kmer, base, kmer_length):
    for idx in range(0, (kmer_length - 1)):
        ascii_kmer[idx] = ascii_kmer[idx + 1]
    ascii_kmer[kmer_length - 1] = base


# checks if aux_kmer is lexicographically smaller than kmer
@cuda.jit(device=True)
def lower(kmer, aux_kmer, kmer_len):
    for idx in range(0, kmer_len):
        if aux_kmer[idx] > kmer[idx]:
            return False
        elif aux_kmer[idx] < kmer[idx]:
            return True
    return False


@cuda.jit(device=True)
def reverse_comp(reverse, kmer_len):
    left = 0
    right = kmer_len - 1
    while left <= right:
        comp_left = complement(reverse[left])
        comp_right = complement(reverse[right])
        reverse[left] = comp_right
        reverse[right] = comp_left
        right -= 1
        left += 1


@cuda.jit(device=True)
def complement(base):
    if base == 1:
        return 4
    elif base == 2:
        return 3
    elif base == 3:
        return 2
    elif base == 4:
        return 1
    else:
        return 5


@cuda.jit(device=True)
def to_decimal_ascii(local_read, seqlen):
    for idx in range(seqlen):
        if local_read[idx] == 1:
            local_read[idx] = 65
        elif local_read[idx] == 2:
            local_read[idx] = 67
        elif local_read[idx] == 3:
            local_read[idx] = 71
        elif local_read[idx] == 4:
            local_read[idx] = 84
        elif local_read[idx] == 5:
            local_read[idx] = 78


@cuda.jit(device=True)
def encode_bases(bases, seqlen):
    for idx in range(0, seqlen):
        if bases[idx] == 5:
            bases[idx] = 1


@cuda.jit(device=True)
def seed_ones(local_read, seqlen):
    for idx in range(seqlen):
        local_read[idx] = 1

@cuda.jit(device=True)
def sort_ping(region_indices,key, regions_num):
    #already sorted
    if regions_num == 1:
        return

    for i in range(1, regions_num):
        copy_kmer(key,region_indices[i], 0, 3)
        j = i - 1
        while j >= 0 and key[2] > region_indices[j][2]:
            copy_kmer(region_indices[j + 1], region_indices[j], 0, 3)
            j -= 1
        # region_indices[j + 1] = key
        copy_kmer(region_indices[j + 1], key, 0, 3)
@cuda.jit(device=True)
def sort_pong(region_indices,key, regions_num):
    #already sorted
    if regions_num == 1:
        return

    for i in range(1, regions_num):
        copy_kmer(key,region_indices[i], 0, 3)
        j = i - 1
        while j >= 0 and key[2] < region_indices[j][2]:
            copy_kmer(region_indices[j + 1], region_indices[j], 0, 3)
            j -= 1
        # region_indices[j + 1] = key
        copy_kmer(region_indices[j + 1], key, 0, 3)
