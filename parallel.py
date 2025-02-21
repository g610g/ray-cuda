import time
import os
import sys
import cudf
import numpy as np
import fastq_parser
import ray
from Bio import SeqIO, Seq
from host.one_sided import entry
from shared_core_correction import *
from numba import cuda
from shared_helpers import (
    test_slice_array,
    to_local_reads,
    back_to_sequence_helper,
    assign_sequence,
)
from utility_helpers.utilities import check_votes
from voting import *
from kmer import *

# import seaborn as sns
# import matplotlib.pyplot as plt

ray.init(dashboard_host="0.0.0.0")


@ray.remote(num_gpus=1)
class GPUActor:
    def ping(self):
        print(
            "GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"])
        )
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


# this task will be run into node that has GPU accelerator
@ray.remote(num_gpus=1)
def gpu_task():
    print("GPU IDs: {}".format(ray.get_runtime_context().get_accelerator_ids()["GPU"]))
    print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))


# start is the starting kmer point
# absolute_start is the read absolute starting point from the 1d read
@cuda.jit(device=True)
def correct_edge_bases(
    idx,
    read,
    kmer,
    bases,
    kmer_len,
    kmer_spectrum,
    threadIdx,
    counter,
    corrected_counter,
):
    current_base = read[idx]
    posibility = 0
    candidate = -1
    for alternative_base in bases:
        if alternative_base != current_base:

            kmer[0] = alternative_base
            candidate_kmer = transform_to_key(kmer, kmer_len)

            if in_spectrum(kmer_spectrum, candidate_kmer):
                posibility += 1
                candidate = alternative_base

    if posibility == 1:
        read[idx] = candidate
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    if posibility == 0:
        corrected_counter[threadIdx][counter] = 10
        counter += 1

    if posibility > 1:
        corrected_counter[threadIdx][counter] = posibility
        counter += 1

    return counter


@ray.remote(num_cpus=1)
def check_corrections(kmers_tracker):
    # overcorrected_num = 0
    for tracker in kmers_tracker:
        for correction in tracker:
            if correction >= 2:
                print(tracker)
                break

    # print(f"number of overcorrected kmers: {overcorrected_num}")


@ray.remote(num_gpus=1, num_cpus=2)
def test_cuda_array_context():
    
    my_arr = np.zeros((512, 21), dtype="uint16")
    my_aux_arr = np.zeros((512, 21), dtype="uint16")

    for idx in range(10):
        my_arr[idx][0] = idx + 1

    dev_arr = cuda.to_device(my_arr)
    dev_aux_arr = cuda.to_device(my_aux_arr)
    tbp = 512
    bpg = my_arr.shape[0] // tbp
    test_slice_array[bpg, tbp](dev_arr, dev_aux_arr, len(my_arr))
    return (dev_arr.copy_to_host(), dev_aux_arr.copy_to_host())


@ray.remote(num_gpus=1, num_cpus=2)
def remote_core_correction(kmer_spectrum, reads_1d, offsets, kmer_len):
    cuda.profile_start()
    start = cuda.event()
    end = cuda.event()
    start.record()

    # transfering necessary data into GPU side
    dev_reads_1d = cuda.to_device(reads_1d)
    dev_kmer_spectrum = cuda.to_device(kmer_spectrum)
    dev_offsets = cuda.to_device(offsets)
    max_votes = cuda.to_device(np.zeros((offsets.shape[0], 100), dtype="int32"))

    # allocating gpu threads
    tbp = 512
    bpg = offsets.shape[0] // tbp

    # invoking the two sided correction kernel
    two_sided_kernel[bpg, tbp](
        dev_kmer_spectrum,
        dev_reads_1d,
        dev_offsets,
        kmer_len,
    )

    # voting refinement is done within the one_sided_kernel
    one_sided_kernel[bpg, tbp](
        dev_kmer_spectrum, dev_reads_1d, dev_offsets, kmer_len, max_votes
    )

    
    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the kernel:  {transfer_time} ms")

    cuda.profile_stop()
    return [
        dev_reads_1d.copy_to_host(),
        max_votes.copy_to_host(),
    ]


# a kernel that brings back the sequences by using the offsets array
# doing this in a kernel since I havent found any cudf methods that transform my jagged array into a segments known as reads
# it has a drawback where if the length of the read is greater than MAXLEN, kernel will produce incorrect results
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


# inefficient operation a lot of things going in here
def md_array_to_rows_as_strings(md_df):

    translation_table = str.maketrans(
        {"1": "A", "2": "C", "3": "G", "4": "T", "5": "N"}
    )
    rows_as_lists = md_df.to_pandas().apply(lambda row: row.tolist(), axis=1)
    non_zero_rows_as_lists = rows_as_lists.apply(lambda lst: [x for x in lst if x != 0])
    rows_as_strings = non_zero_rows_as_lists.apply(lambda lst: "".join(map(str, lst)))

    cudf_rows_as_strings = cudf.Series(rows_as_strings)
    translated_strs = cudf_rows_as_strings.str.translate(translation_table)
    return translated_strs


# a method to turn back string into biopython sequence
# needs to be refactored since runs slowly
@ray.remote(num_cpus=1, num_gpus=1)
def back_2_sequence(reads, offsets):

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

    read_result_df = cudf.DataFrame(dev_reads_result)

    translated_strs = md_array_to_rows_as_strings(read_result_df)
    reads = translated_strs.to_arrow().to_pylist()
    return reads


@ray.remote(num_cpus=1)
def create_sequence_objects(reads_batch, seq_records_batch):
    for read, seq_record in zip(reads_batch, seq_records_batch):
        seq_record.seq = Seq.Seq(read)
    return seq_records_batch


@ray.remote(num_gpus=1)
class GPUPing:
    def ping(self):
        print(ray.get_runtime_context().get_accelerator_ids())


@ray.remote(num_gpus=1)
def ping_resources():
    print(ray.cluster_resources())


if __name__ == "__main__":
    # (arr, aux_arr) = ray.get(test_cuda_array_context.remote())
    # for idx in range(10):
    #     print(aux_arr[idx][0])
    start_time = time.perf_counter()
    usage = "Usage " + sys.argv[0] + " <FASTQ file>"
    if len(sys.argv) != 2:
        print(usage)
        exit(1)
    cpus_detected = int(ray.cluster_resources()["CPU"])

    reads = fastq_parser.parse_fastq_file(sys.argv[1])
    transform_to_string_end_time = time.perf_counter()
    print(
        f"time it takes to convert Seq object into string: {transform_to_string_end_time - start_time}"
    )
    print(f"Length of reads: {len(reads)}")

    kmer_len = 18
    kmer_extract_start_time = time.perf_counter()
    gpu_extractor = KmerExtractorGPU.remote(kmer_len)
    kmer_occurences = ray.get(gpu_extractor.calculate_kmers_multiplicity.remote(reads, 3000000))
    offsets = ray.get(gpu_extractor.get_offsets.remote(reads))
    reads_1d = ray.get(gpu_extractor.transform_reads_2_1d.remote(reads, 3000000))
    kmer_extract_end_time = time.perf_counter()
    print(
        f"time it takes to Extract kmers and transform kmers: {kmer_extract_end_time - kmer_extract_start_time}"
    )
    print(
        f"number of reads is equal to number of rows in the offset:{offsets.shape[0]}"
    )

    # remove unique kmers
    non_unique_kmers = kmer_occurences[kmer_occurences["multiplicity"] > 1]
    print("Done removing unique kmers")
    occurence_data = non_unique_kmers["multiplicity"].to_numpy()
    max_occurence = occurence_data.max()
    print(f"max occurence data: {max_occurence}")

    print(f"kmer spectrum {non_unique_kmers}")
    cutoff_threshold = calculatecutoff_threshold(occurence_data,  max_occurence // 2)
    # testing static cutoff threshold checking if this calculation causes error
    print(f"cutoff threshold: {cutoff_threshold}")

    batch_size = len(offsets) // cpus_detected

    filtered_kmer_df = non_unique_kmers[
        non_unique_kmers["multiplicity"] >= cutoff_threshold
    ]
    kmer_np = filtered_kmer_df.astype("uint64").to_numpy()
    sort_start_time = time.perf_counter()

    sorted_kmer_np = sorted_arr = kmer_np[kmer_np[:, 0].argsort()]
    sort_end_time = time.perf_counter()

    #sorting the kmer spectrum in order to make the search faster with binary search
    print(f"sorting kmer spectrum takes: {sort_end_time - sort_start_time}")
    print(sorted_kmer_np)

    [corrected_reads_array, votes] = ray.get(
        remote_core_correction.remote(sorted_kmer_np, reads_1d, offsets, kmer_len)
    )

    back_sequence_start_time = time.perf_counter()
    corrected_2d_reads_array = ray.get(
        back_to_sequence_helper.remote(corrected_reads_array, offsets)
    )
    back_sequence_end_time = time.perf_counter()
    print(
        f"time it takes to turn reads back: {back_sequence_end_time - back_sequence_start_time}"
    )

    write_file_starttime = time.perf_counter()
    print(corrected_2d_reads_array)
    fastq_data_list = fastq_parser.write_fastq_file('genetic-assets/please.fastq', sys.argv[1], corrected_2d_reads_array)

    write_file_endtime = time.perf_counter()
    print(
        f"time it takes to write reads back to fastq file: {write_file_endtime - write_file_starttime}"
    )

    end_time = time.perf_counter()
    print(f"Elapsed time is {end_time - start_time}")
