import cudf
from numba import cuda
from helpers import in_spectrum, transform_to_key, mark_solids_array, copy_solids
from Bio import Seq
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
def identify_solid_bases(
    local_reads, kmer_len, kmer_spectrum, solids, ascii_kmer, size
):

    for idx in range(0, size + 1):
        copy_kmer(ascii_kmer, local_reads, idx, idx + kmer_len)
        curr_kmer = transform_to_key(ascii_kmer, kmer_len)

        # set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx, idx + kmer_len)


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
):

    identify_solid_bases(local_reads, kmer_len, kmer_spectrum, solids, aux_kmer, size)

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


@ray.remote(num_cpus=1)
def transform_back_to_row_reads(reads, batch_start, batch_end, offsets):

    for batch_idx in range(batch_start, batch_end):
        start, end = offsets[batch_idx][0], offsets[batch_idx][1]


# from jagged array to 2 dimensional array


@ray.remote(num_cpus=1, num_gpus=1)
def back_to_sequence_helper(reads, offsets):

    # find reads max length
    offsets_df = cudf.DataFrame({"start": offsets[:, 0], "end": offsets[:, 1]})
    offsets_df["length"] = offsets_df["end"] - offsets_df["start"]
    max_segment_length = offsets_df["length"].max()

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

    return dev_reads_result.copy_to_host()


@ray.remote(num_cpus=1)
def increment_array(arr):
    for value in arr:
        value += 1
    return arr


# this task is slow because it returns a Sequence object that needs to be serialized
# ray serializes object references and raw data
@ray.remote(num_cpus=1)
def assign_sequence(read_batch, sequence_batch):
    translation_table = str.maketrans({"1": "A", "2": "C", "3": "G", "4": "T"})
    for int_read, sequence in zip(read_batch, sequence_batch):
        non_zeros_int_read = [x for x in int_read if x != 0]
        read_string = "".join(map(str, non_zeros_int_read))
        ascii_read_string = read_string.translate(translation_table)
        if len(ascii_read_string) != 100:
            print(ascii_read_string)
        sequence.seq = Seq.Seq(ascii_read_string)

    return sequence_batch


@cuda.jit
def calculate_reads_solidity(reads, offsets, solids_array, kmer_len, kmer_spectrum):
    threadIdx = cuda.grid(1)
    if threadIdx <= offsets.shape[0]:
        MAX_LEN = 300
        local_reads = cuda.local.array(MAX_LEN, dtype="uint8")
        local_solids = cuda.local.array(MAX_LEN, dtype="int8")
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]

        for idx in range(end - start):
            local_solids[idx] = -1

        start, end = offsets[threadIdx][0], offsets[threadIdx][1]

        # copy global reads into local variable
        for idx in range(end - start):
            local_reads[idx] = reads[idx + start]

        identify_solid_bases(
            local_reads, start, end, kmer_len, kmer_spectrum, local_solids
        )
        copy_solids(threadIdx, local_solids, solids_array)


@cuda.jit
def back_sequence_kernel(reads, offsets, reads_result):
    threadIdx = cuda.grid(1)
    MAX_LEN = 300
    local_reads = cuda.local.array(MAX_LEN, dtype="uint8")
    if threadIdx < offsets.shape[0]:
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]
        seqlen = end - start

        to_local_reads(reads, local_reads, start, end)
        to_decimal_ascii(local_reads, seqlen)

        # copy the assigned read for this thread into the 2d reads_result
        for idx in range(end - start):
            reads_result[threadIdx][idx] = local_reads[idx]


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
    local_read,
    aux_kmer,
    kmer_spectrum,
    alternative_base,
    spos,
    distance,
    read_len,
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
        forward_base(aux_kmer, local_read[offset + idx + kmer_length - 1], kmer_length)
        aux_kmer[counter] = alternative_base
        transformed_alternative_kmer = transform_to_key(aux_kmer, kmer_length)
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
):
    ipos = target_pos - 1
    if ipos < 0 or distance <= 0:
        return True
    spos = max(0, ipos - distance)
    counter = 1
    idx = ipos
    while idx >= spos:
        copy_kmer(aux_kmer, local_read, idx, idx + kmer_length)
        # backward_base(aux_kmer, local_read[idx], kmer_length)
        aux_kmer[counter] = alternative_base
        candidate = transform_to_key(aux_kmer, kmer_length)
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


@cuda.jit(device=True)
def test_copying(arr1):
    for idx in range(5):
        arr1[idx] = idx + 1


@cuda.jit()
def test_slice_array(arr, aux_arr_storage, arr_len):
    threadIdx = cuda.grid(1)
    if threadIdx <= arr_len:
        local_kmer = cuda.local.array(21, dtype="uint8")
        


@cuda.jit(device=True)
def copy_kmer(aux_kmer, local_read, start, end):
    for i in range(start, end):
        aux_kmer[i - start] = local_read[i]


@cuda.jit(device=True)
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

    # assign back the original base
    ascii_kmer[pos] = original_base

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
    for idx in range(0, kmer_length):
        if idx == (kmer_length - 1):
            ascii_kmer[idx] = base
        else:
            ascii_kmer[idx] = ascii_kmer[idx + 1]

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
        else:
            local_read[idx] = 0

