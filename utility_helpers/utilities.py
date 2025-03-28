from helpers import to_array_kmer, transform_to_key
import ray
from numba import cuda
from shared_helpers import copy_kmer, lower, reverse_comp
import fastq_parser

def get_filename_without_extension(file_path) -> str:
    if file_path.endswith((".fastq")):
        return file_path.split(".fastq")[0]
    elif file_path.endswith((".fq")):
        return file_path.split(".fq")[0]

    # NOTE:: fix this here
    return ""

def partition_reads(num_partition, reads_len)->list:
    bounds = []
    start = 0
    reads_per_task = reads_len // num_partition 
    remainder = reads_len % num_partition
    for i in range(5):
        extra = 1 if i < remainder else 0  # Distribute remainder to first GPUs
        end = start + reads_per_task + extra
        bounds.append([start, end, end - start])
        start = end
    return bounds
@ray.remote(num_cpus=1)
def calculate_solids(solids):
    for solid in solids:
        solids_count = 0
        for base in solid:
            if base == -1:
                solids_count += 1

        if solids_count > 0:
            print(solids_count)


@ray.remote(num_cpus=1)
def check_onesided_corrections(corrections):
    trusted_counts = 0
    for correction in corrections:
        if correction < 0:
            trusted_counts += 1

    print(f"{trusted_counts} of reads have solid bases out of {len(corrections)} reads")


@ray.remote(num_cpus=1)
def check_twosided_corrections(corrections):
    corrections_count = 0
    for correction in corrections:
        if correction < 0:
            corrections_count += 1
            continue
        corrections_count += correction
    print(
        f"{corrections_count} conducts two sided correction out of {len(corrections)}"
    )


@ray.remote(num_cpus=1)
def check_votes(votes):
    has_votes = 0
    for vote in votes:
        for item in vote:
            if item > 0:
                has_votes += 1
                break

    print(has_votes)


@cuda.jit(device=True)
def test_return_value(val, iter, bases):
    sum = 0
    for base in bases:
        sum += base
    return sum


@cuda.jit()
def reverse_comp_kmer(dev_kmers, kmer_length):
    MAX_KMER_LEN = 18
    threadIdx = cuda.grid(1)
    if threadIdx < dev_kmers.shape[0]:
        rep = cuda.local.array(MAX_KMER_LEN, dtype="uint8")
        km = cuda.local.array(MAX_KMER_LEN, dtype="uint8")
        to_array_kmer(km, kmer_length, dev_kmers[threadIdx][0])
        copy_kmer(rep, km, 0, kmer_length)
        reverse_comp(rep, kmer_length)
        if lower(rep, km, kmer_length):
            rev_comp_km = transform_to_key(km, kmer_length)
            dev_kmers[threadIdx][0] = rev_comp_km
        else:
            rev_comp_km = transform_to_key(rep, kmer_length)
            dev_kmers[threadIdx][0] = rev_comp_km


# benchmark for how many reads got seeded by ones within one sided kernel
@ray.remote(num_cpus=1)
def bench_corrections(corrections, seqlen):
    ones = 0
    for correction in corrections:
        counts = 0
        for element in correction:
            if element == 65:
                counts += 1
        if counts == seqlen:
            ones += 1
    print(f"{ones} reads that all elements are one out of {len(corrections)} reads")


@ray.remote(num_cpus=1)
def check_solids(solids, seqlen):
    ones = 0
    for solid in solids:
        count = 0
        for element in solid:
            if element == 1:
                count += 1
        if count == seqlen:
            ones += 1
            # print(solid)
    print(f"{ones} solid reads out of  {solids.shape[0]} reads")


@ray.remote(num_cpus=1)
def check_uncorrected(solids):
    ones = 0
    for solid in solids:
        for element in solid:
            if element == 0:
                ones += 1
                break
    print(f"{ones} uncorrected reads out of {solids.shape[0]} reads")


@ray.remote(num_cpus=1)
def check_has_corrected(corrected):
    corrected_count = 0
    for correction in corrected:
        if correction == 1:
            corrected_count += 1
    print(
        f"{corrected_count} number of reads that has been corrected out of  {len(corrected)} reads"
    )
@ray.remote(num_cpus=1)
def calculate_non_solids(solids, seqlen):
    unsolid = 0
    for solid in solids:
        for element in solid:
            if element < 0 or element == 0:
                unsolid += 1
                break
    print(
        f"{unsolid} number of reads that is not solid out of {len(solids)} reads"
    )

@ray.remote(num_cpus=1)
def write_fastq_file(output_filename, src_filename, bound, reads):
    print(bound)
    print(reads)
    fastq_parser.write_fastq_file(
        output_filename, src_filename, reads, bound[0]
    )
