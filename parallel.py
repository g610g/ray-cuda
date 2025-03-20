import time
import os
import sys
import cudf
import numpy as np
import fastq_parser
import kmer
import ray
from Bio import Seq
from shared_core_correction import *
from shared_helpers import (
    sort_ping,
    sort_pong,
    to_local_reads,
    back_to_sequence_helper,
)
from voting import *
from kmer import *
from utility_helpers.utilities import *

ray.init(dashboard_host="0.0.0.0")
# os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import cuda


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
def remote_core_correction(
    kmer_spectrum, reads_2d, offsets, kmer_len, last_end_idx, batch_size
):
    cuda.profile_start()
    start = cuda.event()
    end = cuda.event()
    start.record()
    print(f"offset shape: {offsets.shape}")
    print(f"offset dtype: {offsets.dtype}")
    print(f"reads dtype: {reads_2d.dtype}")
    print(f"Kmer spectrum: {kmer_spectrum}")
    # transfering necessary data into GPU side
    MAX_READ_LENGTH = 400
    dev_reads_2d = cuda.to_device(reads_2d)
    dev_kmer_spectrum = cuda.to_device(kmer_spectrum)
    dev_offsets = cuda.to_device(offsets)
    max_votes = cuda.to_device(np.zeros((offsets.shape[0], 100), dtype="int32"))
    dev_reads_corrected_2d = cuda.to_device(
        np.zeros((offsets.shape[0], MAX_READ_LENGTH), dtype="uint8")
    )
    solids = cuda.to_device(np.zeros((offsets.shape[0], MAX_READ_LENGTH), dtype="int8"))
    # allocating gpu threads
    # bpg = math.ceil(offsets.shape[0] // tpb)
    tpb = 512
    bpg = (offsets.shape[0] + tpb) // tpb

    one_sided_kernel[bpg, tpb](
        dev_kmer_spectrum,
        dev_reads_2d,
        dev_offsets,
        kmer_len,
        max_votes,
        dev_reads_corrected_2d,
        solids,
    )

    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the kernel:  {transfer_time} ms")

    cuda.profile_stop()
    return dev_reads_corrected_2d.copy_to_host(), solids.copy_to_host()


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


@cuda.jit()
def kernel_test(dev_arr, dev_len):
    threadIdx = cuda.grid(1)
    if threadIdx < dev_arr.shape[0]:
        region_indices = cuda.local.array((10, 3), dtype="uint32")
        key = cuda.local.array(3, dtype="uint32")
        # Initialize the array elements individually
        # region_indices[3, 0] = 1
        # region_indices[3, 1] = 3
        # region_indices[3, 2] = 2
        #
        # region_indices[2, 0] = 5
        # region_indices[2, 1] = 10
        # region_indices[2, 2] = 5
        #
        # region_indices[1, 0] = 10
        # region_indices[1, 1] = 25
        # region_indices[1, 2] = 15
        #
        # region_indices[0, 0] = 30
        # region_indices[0, 1] = 50
        # region_indices[0, 2] = 20
        # regions_num = 4
        # sort_pong(region_indices, key, regions_num)
        # for i in range(regions_num):
        #     for j in range(3):
        #         dev_arr[threadIdx][i][j] = region_indices[i][j]
        dev_len[threadIdx][0] = len(region_indices) // 2


# NOTE:: we will create dataframe and group them then we return the grouped kmers that will be stored for each gpus
@ray.remote(num_gpus=0.5, num_cpus=1)
def combine_kmers(kmers):
    kmers = np.concatenate(kmers, axis=0)
    kmers = cudf.DataFrame({"canonical": kmers[:, 0], "multiplicity": kmers[:, 1]})
    grouped_kmers = kmers.groupby("canonical").sum().reset_index()
    print(grouped_kmers)
    return grouped_kmers


@ray.remote(num_gpus=1, num_cpus=1)
def test():
    arr = np.zeros((10, 10, 3), dtype="uint32")
    length = np.zeros((10, 2), dtype="uint32")
    dev_arr = cuda.to_device(arr)
    dev_len = cuda.to_device(length)
    tpb = len(arr)
    bpg = (len(arr) + tpb) // tpb

    kernel_test[bpg, tpb](dev_arr, dev_len)
    return dev_len.copy_to_host()


if __name__ == "__main__":
    print(ray.get(test.remote()))
    start_time = time.perf_counter()
    usage = "Usage " + sys.argv[0] + " <FASTQ file> <FASTQ READS COUNT>"
    if len(sys.argv) != 3:
        print(usage)
        exit(1)
    cpus_detected = int(ray.cluster_resources()["CPU"])
    gpus_detected = int(ray.cluster_resources()["GPU"])
    print(f"number of gpus detected: {gpus_detected}")
    kmer_len = 16
    parse_reads_starttime = time.perf_counter()
    reads_len = int(sys.argv[2])
    print(f"length of reads: {reads_len}")
    

    transform_to_string_end_time = time.perf_counter()
    print(
        f"time it takes to convert Seq object into string: {transform_to_string_end_time - start_time}"
    )
    kmer_extract_start_time = time.perf_counter()
    kmer_actors = []
    reads_per_gpu = reads_len // gpus_detected
    remainder = reads_len % gpus_detected
    start = 0
    start_end = []
    for i in range(gpus_detected):
        extra = 1 if i < remainder else 0  # Distribute remainder to first GPUs
        end = start + reads_per_gpu + extra
        start_end.append([start, end, end - start])
        start = end

    print(reads_len)
    print(start_end)
    for bound in start_end:
        kmer_actors.append(
            KmerExtractorGPU.remote(kmer_len, bound, sys.argv[1])
        )
        print(bound)

    kmer_extract_references = []
    offsets_extract_references = []
    reads_2d_references = []

    ray.get([kmer_actor.extract_reads.remote() for kmer_actor in kmer_actors])
    parse_reads_endtime = time.perf_counter()
    print(
        f"Time it takes to parse reads {parse_reads_endtime - parse_reads_starttime}"
    )
    for kmer_actor in kmer_actors:
        kmer_extract_references.append(
            kmer_actor.calculate_kmers_multiplicity.remote(100000)
        )
    kmers = ray.get(kmer_extract_references)
    # print(kmers)
    for kmer_actor in kmer_actors:
        offsets_extract_references.append(kmer_actor.get_offsets.remote())

    ray.get(offsets_extract_references)

    for kmer_actor in kmer_actors:
        reads_2d_references.append(kmer_actor.transform_reads_2_1d.remote(100000))

    ray.get(reads_2d_references)
    kmer_occurences = ray.get(kmer_actors[0].combine_kmers.remote(kmers))
    print(kmer_occurences)
    kmer_extract_end_time = time.perf_counter()
    print(
        f"time it takes to Extract kmers and transform kmers: {kmer_extract_end_time - kmer_extract_start_time}"
    )

    # print(
    #     f"number of reads is equal to number of rows in the offset:{offsets.shape[0]}"
    # )

    # offsets_df = cudf.DataFrame({"start": offsets[:, 0], "end": offsets[:, 1]})
    # offsets_df["length"] = offsets_df["end"] - offsets_df["start"]
    # max_segment_length = offsets_df["length"].max()
    # print(f"max length is {max_segment_length}")
    # remove unique kmers
    non_unique_kmers = kmer_occurences[kmer_occurences["multiplicity"] > 1]
    print("Done removing unique kmers")
    occurence_data = non_unique_kmers["multiplicity"].to_numpy().astype(np.uint64)
    max_occurence = occurence_data.max()
    print(f"max occurence data: {max_occurence}")

    print(f"Non unique kmers {non_unique_kmers}")
    cutoff_threshold = calculatecutoff_threshold(occurence_data, max_occurence)
    # cutoff_threshold = 10
    print(f"cutoff threshold: {cutoff_threshold}")

    # batch_size = len(offsets) // cpus_detected
    filtered_kmer_df = non_unique_kmers[
        non_unique_kmers["multiplicity"] >= cutoff_threshold
    ]

    kmer_np = filtered_kmer_df.astype("uint64").to_numpy()
    print(kmer_np)
    print(f"Number of trusted kmers: {kmer_np.shape[0]}")

    sort_start_time = time.perf_counter()

    sorted_kmer_np = kmer_np[kmer_np[:, 0].argsort()]
    sorted_by_occurence = kmer_np[kmer_np[:, 1].argsort()[::-1]]
    sort_end_time = time.perf_counter()

    # print(f"reads 1d length: {len(reads_2d)}")
    # sorting the kmer spectrum in order to make the search faster with binary search
    print(f"sorted by occurence {sorted_by_occurence[:100]}")
    print(f"sorted by kmer {sorted_kmer_np[:100]}")
    print(f"sorting kmer spectrum takes: {sort_end_time - sort_start_time}")

    # NOTE::correcting by batch is not used currently
    # we will try correcting by batch
    sorted_kmer_np_reference = ray.put(sorted_kmer_np)
    correction_batch_size = 1000000
    correction_result = []
    last_end_idx = 0
    ray.get(
        [
            kmer_actor.update_spectrum.remote(sorted_kmer_np_reference)
            for kmer_actor in kmer_actors
        ]
    )
    ray.get([kmer_actor.correct_reads.remote() for kmer_actor in kmer_actors])
    back_sequence_start_time = time.perf_counter()
    corrected_reads = ray.get(
        [kmer_actor.back_to_sequence_helper.remote() for kmer_actor in kmer_actors]
    )
    corrected_2d_reads_array = np.concatenate(corrected_reads)
    print(f"corrected reads array {corrected_2d_reads_array}")
    print(corrected_2d_reads_array.dtype)

    # print(solids_host[1])
    # print(solids_host[2])
    # ray.get(
    #     [
    #         check_solids.remote(solids_host[idx : idx + batch_size], 100)
    #         for idx in range(0, len(solids_host), batch_size)
    #     ]
    # )
    # ray.get(
    #     [
    #         calculate_non_solids.remote(solids_host[idx : idx + batch_size], 100)
    #         for idx in range(0, len(solids_host), batch_size)
    #     ]
    # )
    # ray.get(
    #     [
    #         check_uncorrected.remote(solids_host[idx : idx + batch_size])
    #         for idx in range(0, len(solids_host), batch_size)
    #     ]
    # )
    back_sequence_end_time = time.perf_counter()

    write_file_starttime = time.perf_counter()

    filename = get_filename_without_extension(sys.argv[1])
    output_filename = filename + "GPUMUSKET.fastq"
    print(output_filename)
    fastq_data_list = fastq_parser.write_fastq_file_v2(
        output_filename, sys.argv[1], corrected_2d_reads_array
    )

    write_file_endtime = time.perf_counter()
    print(
        f"time it takes to write reads back to fastq file: {write_file_endtime - write_file_starttime}"
    )

    end_time = time.perf_counter()
    print(f"Elapsed time is {end_time - start_time}")
