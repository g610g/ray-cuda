from threading import local
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


# computes the kmer that are affected for the correction and increment the correction count on each computed kmer
@cuda.jit(device=True)
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


@cuda.jit(device=True)
def identify_solid_bases(local_reads, start, end, kmer_len, kmer_spectrum, solids):

    for idx in range(0, (end - start) - (kmer_len - 1)):
        ascii_kmer = local_reads[idx : idx + kmer_len]

        curr_kmer = transform_to_key(ascii_kmer, kmer_len)

        # set the bases as solids
        if in_spectrum(kmer_spectrum, curr_kmer):
            mark_solids_array(solids, idx, (idx + kmer_len))


@cuda.jit(device=True)
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
    translation_table = str.maketrans(
        {"1": "A", "2": "C", "3": "G", "4": "T", "5": "N"}
    )
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
        to_local_reads(reads, local_reads, start, end)

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

#lookahead validation of succeeding kmers
@cuda.jit(device=True)
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

#lookahead validation of preceeding kmers
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
