import time
import os
import gc
import psutil
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_record_ref_creation_sites"]="1"
import sys
import cudf
import numpy as np
import fastq_parser
import ray
from shared_core_correction import *
from kmer import *
from utility_helpers.utilities import *
# ray.init(dashboard_host="0.0.0.0", object_store_memory=0.7 * psutil.virtual_memory().total)
ray.init()

# RAY_DEDUP_LOGS=0
from numba import cuda

@ray.remote(num_gpus=1, num_cpus=2)
def remote_core_correction(kmer_spectrum, reads_2d, offsets, kmer_len):
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
    print("done transferring reads into device")
    dev_kmer_spectrum = cuda.to_device(kmer_spectrum)
    dev_offsets = cuda.to_device(offsets)
    dev_reads_corrected_2d = cuda.to_device(
        np.zeros((offsets.shape[0], MAX_READ_LENGTH), dtype="uint8")
    )
    tpb = 512
    bpg = (offsets.shape[0] + tpb) // tpb

    one_sided_kernel[bpg, tpb](
        dev_kmer_spectrum,
        dev_reads_2d,
        dev_offsets,
        kmer_len,
        dev_reads_corrected_2d,
    )

    end.record()
    end.synchronize()
    transfer_time = cuda.event_elapsed_time(start, end)
    print(f"execution time of the kernel:  {transfer_time} ms")

    cuda.profile_stop()
    return dev_reads_corrected_2d.copy_to_host()



if __name__ == "__main__":
    start_time = time.perf_counter()
    usage = "Usage " + sys.argv[0] + " <FASTQ file> <FASTQ READS COUNT>"
    if len(sys.argv) != 3:
        print(usage)
        exit(1)
    cpus_detected = int(ray.cluster_resources()["CPU"])
    gpus_detected = int(ray.cluster_resources()["GPU"])
    gpus_detected = 3
    # hasone_gpu = gpus_detected == 1
    print(f"number of gpus detected: {gpus_detected}")
    kmer_len = 16
    parse_reads_starttime = time.perf_counter()
    reads_len = int(sys.argv[2])
    # print(f"length of reads: {reads_len}")

    transform_to_string_end_time = time.perf_counter()
    print(
        f"time it takes to convert Seq object into string: {transform_to_string_end_time - start_time}"
    )
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

    # print(reads_len)
    # print(start_end)
    for bound in start_end:
        kmer_actors.append(KmerExtractorGPU.remote(kmer_len, bound, sys.argv[1]))

    kmer_extract_references = []
    offsets_extract_references = []
    reads_2d_references = []
    reads = []
    ray.get([kmer_actor.extract_reads.remote() for kmer_actor in kmer_actors])
    parse_reads_endtime = time.perf_counter()
    print(f"Time it takes to parse reads {parse_reads_endtime - parse_reads_starttime}")
    kmer_extract_start_time = time.perf_counter()
    for kmer_actor in kmer_actors:
        kmer_extract_references.append(
            kmer_actor.calculate_kmers_multiplicity.remote(1000000)
        )
    kmers = ray.get(kmer_extract_references)
    del kmer_extract_references

    for kmer_actor in kmer_actors:
        offsets_extract_references.append(kmer_actor.get_offsets.remote())

    ray.get(offsets_extract_references)
    del offsets_extract_references
    for kmer_actor in kmer_actors:
        reads_2d_references.append(kmer_actor.transform_reads_2_1d.remote(2000000))

    ray.get(reads_2d_references)
    del reads_2d_references
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
    print(f"cutoff threshold: {cutoff_threshold}")

    filtered_kmer_df = non_unique_kmers[
        non_unique_kmers["multiplicity"] >= cutoff_threshold
    ]

    kmer_np = filtered_kmer_df.astype("uint64").to_numpy()

    # freeing GPU memory
    del non_unique_kmers
    del occurence_data
    del kmer_occurences
    del kmers
    del filtered_kmer_df

    print(kmer_np)
    print(f"Number of trusted kmers: {kmer_np.shape[0]}")

    sort_start_time = time.perf_counter()

    sorted_kmer_np = kmer_np[kmer_np[:, 0].argsort()]
    sort_end_time = time.perf_counter()

    print(f"sorting kmer spectrum takes: {sort_end_time - sort_start_time}")
    sorted_kmer_np_reference = ray.put(sorted_kmer_np)

    del sorted_kmer_np
    del kmer_np
    ray.get(
        [
            kmer_actor.update_spectrum.remote(sorted_kmer_np_reference)
            for kmer_actor in kmer_actors
        ]
    )
    ray.get([kmer_actor.correct_reads.remote() for kmer_actor in kmer_actors])

    del sorted_kmer_np_reference

    print("done correcting")
    back_sequence_start_time = time.perf_counter()
    ray.get([kmer_actor.back_to_sequence_helper.remote() for kmer_actor in kmer_actors])
    print("Correction and back to sequence is now done")
    back_sequence_end_time = time.perf_counter()

    write_file_starttime = time.perf_counter()

    filename = get_filename_without_extension(sys.argv[1])
    output_filename = filename + "GPUMUSKET.fastq"
    print(output_filename)

    write_references = []
    output_files = []
    print("retrieving large amount of data")
    reads_actors = []
    if gpus_detected == 1:
        reads_actors = kmer_actors[0].get_actor_reads.remote()
    else:
        reads_actors = [kmer_actor.get_actor_reads.remote() for kmer_actor in kmer_actors]

    del kmer_actors
    bounds = partition_reads(gpus_detected, reads_len)
    num = 1
    # we write reads into multiple file and then combine them afterwards
    for (bound, read)in zip(bounds, reads_actors):
        local_filename = filename + "GPUMUSKET" + str(num) + ".fastq"
        output_files.append(local_filename)
        write_references.append(
            write_fastq_file.remote(
                local_filename, sys.argv[1], bound, read
            )
        )
        num += 1
    ray.get(write_references)
    print("Done writing to different files")
    fastq_parser.combine_files(output_files, output_filename)
    print("Done combining different files")

    write_file_endtime = time.perf_counter()
    print(
        f"time it takes to write reads back to fastq file: {write_file_endtime - write_file_starttime}"
    )

    end_time = time.perf_counter()
    print(f"Elapsed time is {end_time - start_time}")
