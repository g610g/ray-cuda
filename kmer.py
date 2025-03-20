from typing import final
from unittest import result
import ray
import math
import numpy as np
import cudf
import fastq_parser
from numba import cuda
from shared_core_correction import one_sided_kernel
from shared_helpers import  back_sequence_kernel
from utility_helpers.utilities import reverse_comp_kmer



@ray.remote(num_gpus=1, num_cpus=4)
class KmerExtractorGPU:
    def __init__(self, kmer_length, bounds, fastq_filepath):
        self.kmer_length = kmer_length
        self.translation_table = str.maketrans(
            {
                "A": "1",
                "C": "2",
                "G": "3",
                "T": "4",
                "N": "5",
                "R": "5",
                "M": "5",
                "K": "5",
                "S": "5",
                "W": "5",
                "Y": "5",
            }
        )
        self.fastq_filepath = fastq_filepath
        self.bound = bounds
        self.reads = []
        self.spectrum = []
        self.offsets = []
        self.corrected_reads = []

    def update_spectrum(self, spectrum):
        self.spectrum = spectrum
    def extract_reads(self):
        if len(self.bound) == 0:
            print("Bound is not set. Add better error handling")
            return
        self.reads = fastq_parser.parse_fastq_foreach(self.fastq_filepath, self.bound[0], self.bound[2])
        
    def create_kmer_df(self, reads):
        read_df = cudf.Series(reads)
        kmers = read_df.str.character_ngrams(self.kmer_length, True)
        exploded_kmers = kmers.explode()
        return exploded_kmers.value_counts()

    def get_offsets(self):
        read_s = cudf.Series(self.reads, name="reads")
        read_df = read_s.to_frame()
        str_lens = read_df["reads"].str.len()
        end_indices = str_lens.cumsum()
        start_indices = end_indices.shift(1, fill_value=0)
        offsets = cudf.DataFrame(
            {"start_indices": start_indices, "end_indices": end_indices}
        ).to_numpy()
        self.offsets = offsets
        return

    def transform_reads_2_1d_batch(self, batch_size):
        result = []
        read_s = cudf.Series(self.reads, name="reads")
        max_length = read_s.str.len().max()

        for i in range(0, len(self.reads), batch_size):
            read_s = cudf.Series(self.reads[i : i + batch_size], name="reads")
            padded_reads = read_s.str.pad(width=max_length, side="right", fillchar="0")
            transformed = (
                padded_reads.str.findall(".")
                .explode()
                .str.translate(self.translation_table)
                .astype("uint8")
            )
            result.append(transformed.to_numpy().reshape(len(read_s), max_length))

        concatenated = np.concatenate(result).astype("uint8")
        # print(concatenated)
        return concatenated

    def transform_reads_2_1d(self, batch_size):
        if len(self.reads) > batch_size:
            self.reads = self.transform_reads_2_1d_batch(batch_size)
            return
        read_s = cudf.Series(self.reads, name="reads")
        max_length = read_s.str.len().max()
        padded_reads = read_s.str.pad(width=max_length, side="right", fillchar="0")
        transformed = (
            padded_reads.str.findall(".")
            .explode()
            .str.translate(self.translation_table)
            .astype("uint8")
        )
        self.reads = transformed.to_numpy().reshape(len(self.reads), max_length)

    def get_read_lens(self, reads):
        read_df = cudf.DataFrame({"reads": reads})
        return read_df["reads"].str.len()

    def check_rev_comp_kmer(self, kmer_df):
        # (refactor this type conversions)
        kmer_np = kmer_df.astype("uint64").to_numpy().astype(np.uint64)
        dev_kmers = cuda.to_device(kmer_np)
        dev_kmer_array = cuda.to_device(
            np.zeros((kmer_np.shape[0], self.kmer_length), dtype="uint8")
        )
        tbp = 1024
        # bpg = math.ceil(kmer_np.shape[0] / tbp)
        bpg = (kmer_np.shape[0] + tbp) // tbp
        reverse_comp_kmer[bpg, tbp](dev_kmers, self.kmer_length)
        kmers = dev_kmers.copy_to_host()
        return [kmers, dev_kmer_array.copy_to_host()]

    # store the dataframe as state of this worker to be used by downstream tasks
    def calculate_kmers_multiplicity_batch(self, batch_size):
        all_results = []
        for i in range(0, len(self.reads), batch_size):

            read_s = cudf.Series(self.reads[i : i + batch_size], name="reads")
            read_df = read_s.to_frame()

            replaced_df = read_df["reads"].str.replace(r"[NWKSYMR]", "A")
            # print(replaced_df)
            read_df["translated"] = replaced_df.str.translate(self.translation_table)
            ngram_kmers = read_df["translated"].str.character_ngrams(
                self.kmer_length, True
            )
            exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
            # unique_chars = set("".join(exploded_ngrams.to_pandas().astype(str)))
            # print(unique_chars)
            numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
            result_frame = numeric_ngrams.value_counts().reset_index()
            all_results.append(result_frame)

        concat_result = (
            cudf.concat(all_results, ignore_index=True)
            .groupby("translated")
            .sum()
            .reset_index()
        )
        # print(concat_result)
        concat_result.columns = ["translated", "multiplicity"]
        concat_result["multiplicity"] = concat_result["multiplicity"].clip(upper=255)
        # print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        # print(f"final result shape is: {concat_result.shape}")
        # print(f"Kmers before calculating canonical kmers: {concat_result}")
        [kmers_np, canonical_kmers] = self.check_rev_comp_kmer(concat_result)

        final_kmers = (
            cudf.DataFrame(
                {"canonical": kmers_np[:, 0], "multiplicity": kmers_np[:, 1]}
            )
            .groupby("canonical")
            .sum()
            .reset_index()
        )
        final_kmers["multiplicity"] = final_kmers["multiplicity"].clip(upper=255)
        # print(f"Kmers after calculating canonical kmers: {final_kmers}")
        return final_kmers.to_numpy().astype("uint64", copy=False)

    # lets set arbitrary amount of batch size for canonical kmer calculation

    def calculate_kmers_multiplicity(self, batch_size):

        if len(self.reads) > batch_size:
            return self.calculate_kmers_multiplicity_batch(batch_size)

        read_s = cudf.Series(self.reads, name="reads")
        read_df = read_s.to_frame()

        replaced_df = read_df["reads"].str.replace(r"[NWKSYMR]", "A")
        read_df["translated"] = replaced_df.str.translate(self.translation_table)

        # computes canonical kmers
        ngram_kmers = read_df["translated"].str.character_ngrams(self.kmer_length, True)
        exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)

        numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
        result_frame = numeric_ngrams.value_counts().reset_index()

        result_frame.columns = ["translated", "multiplicity"]
        print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        # print(f"Kmers before calculating canonical kmers: {result_frame}")
        # we do this by batch
        [kmers_np, _] = self.check_rev_comp_kmer(result_frame)

        final_kmers = (
            cudf.DataFrame(
                {"canonical": kmers_np[:, 0], "multiplicity": kmers_np[:, 1]}
            )
            .groupby("canonical")
            .sum()
            .reset_index()
        )
        final_kmers["multiplicity"] = final_kmers["multiplicity"].clip(upper=255)
        # print(f"Kmers after calculating canonical kmers: {final_kmers}")
        return final_kmers.to_numpy().astype("uint64", copy=False)

    def combine_kmers(self, kmers):
        kmers = np.concatenate(kmers)
        print(f"Kmers within combine: {kmers}")
        kmers = cudf.DataFrame({"canonical": kmers[:, 0], "multiplicity": kmers[:, 1]})
        print(f"Kmer dataframe within combine:{kmers}")
        grouped_kmers = kmers.groupby("canonical").sum().reset_index()
        grouped_kmers['multiplicity'] = grouped_kmers['multiplicity'].clip(upper=255)
        return grouped_kmers
    def correct_reads(self):
        cuda.profile_start()
        start = cuda.event()
        end = cuda.event()
        start.record()
        print(f"offset shape: {self.offsets.shape}")
        print(f"offset dtype: {self.offsets.dtype}")
        print(f"reads dtype: {self.reads.dtype}")
        print(f"Kmer spectrum: {self.spectrum}")
        # transfering necessary data into GPU side
        MAX_READ_LENGTH = 400
        dev_reads_2d = cuda.to_device(self.reads)
        dev_kmer_spectrum = cuda.to_device(self.spectrum)
        dev_offsets = cuda.to_device(self.offsets)
        max_votes = cuda.to_device(
            np.zeros((self.offsets.shape[0], 100), dtype="int32")
        )
        dev_reads_corrected_2d = cuda.to_device(
            np.zeros((self.offsets.shape[0], MAX_READ_LENGTH), dtype="uint8")
        )
        solids = cuda.to_device(
            np.zeros((self.offsets.shape[0], MAX_READ_LENGTH), dtype="int8")
        )
        # allocating gpu threads
        # bpg = math.ceil(offsets.shape[0] // tpb)
        tpb = 256
        bpg = (self.offsets.shape[0] + tpb) // tpb

        one_sided_kernel[bpg, tpb](
            dev_kmer_spectrum,
            dev_reads_2d,
            dev_offsets,
            self.kmer_length,
            max_votes,
            dev_reads_corrected_2d,
            solids,
        )

        end.record()
        end.synchronize()
        transfer_time = cuda.event_elapsed_time(start, end)
        print(f"execution time of the kernel:  {transfer_time} ms")
        self.corrected_reads = dev_reads_corrected_2d.copy_to_host()
        cuda.profile_stop()
    def back_to_sequence_helper(self):
        # find reads max length
        offsets_df = cudf.DataFrame({"start": self.offsets[:, 0], "end": self.offsets[:, 1]})
        offsets_df["length"] = offsets_df["end"] - offsets_df["start"]
        max_segment_length = offsets_df["length"].max()
        print(f"max segment length: {max_segment_length}")
        cuda.profile_start()
        start = cuda.event()
        end = cuda.event()
        start.record()

        dev_reads = cuda.to_device(self.corrected_reads)
        dev_offsets = cuda.to_device(self.offsets)
        tpb = 1024
        bpg = (self.offsets.shape[0] + tpb) // tpb

        back_sequence_kernel[bpg, tpb](dev_reads, dev_offsets)

        end.record()
        end.synchronize()
        transfer_time = cuda.event_elapsed_time(start, end)
        print(f"execution time of the back to sequence kernel:  {transfer_time} ms")
        cuda.profile_stop()

        return dev_reads.copy_to_host()


# @ray.remote(num_gpus=0.5, num_cpus=1)
# class GPUActor:
#     def __init__(self, gpuExtractor):
#         self.spectrum = []
#         self.gpuExtractor = gpuExtractor
#     def run(self):
#         spectrumRef = self.gpuExtractor.calculate_kmers_multiplicity.remote(150000)
#         offsetsRef = self.gpuExtractor.get_offsets.remote()
#         transformRef = self.gpuExtractor.transform_reads_2_1d.remote(150000)
#         ray.get(offsetsRef) 
#         ray.get(transformRef) 
#         return ray.get(spectrumRef)
# TODO::refactor
def calculatecutoff_threshold(occurence_data, bin):

    hist_vals, bin_edges = np.histogram(occurence_data, bins=int(bin))

    print((hist_vals[:]))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    print(bin_centers[:])

    valley_index = 0

    for idx in range(valley_index + 1, len(hist_vals)):
        if hist_vals[idx] > hist_vals[valley_index]:
            break
        valley_index = idx
    peak_index = valley_index
    for idx in range(peak_index + 1, len(hist_vals)):
        if hist_vals[idx] > hist_vals[peak_index]:
            peak_index = idx

    min_density_idx = valley_index
    # we will find  the lowest density between valley_idx and peak_idx
    for idx in range(valley_index, peak_index + 1):
        if hist_vals[idx] <= hist_vals[min_density_idx]:
            min_density_idx = idx
    return math.ceil(bin_centers[min_density_idx])
