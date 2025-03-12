from typing import final
from unittest import result
import ray
import math
import numpy as np
import cudf
from numba import cuda
from shared_helpers import reverse_comp
from utility_helpers.utilities import reverse_comp_kmer


@ray.remote(num_gpus=1, num_cpus=1)
class KmerExtractorGPU:
    def __init__(self, kmer_length):
        self.kmer_length = kmer_length
        self.translation_table = str.maketrans(
            {"A": "1", "C": "2", "G": "3", "T": "4", "N": "5"}
        )

    def create_kmer_df(self, reads):
        read_df = cudf.Series(reads)
        kmers = read_df.str.character_ngrams(self.kmer_length, True)
        exploded_kmers = kmers.explode()
        return exploded_kmers.value_counts()

    def get_offsets(self, reads):
        read_s = cudf.Series(reads, name="reads")
        read_df = read_s.to_frame()
        str_lens = read_df["reads"].str.len()
        end_indices = str_lens.cumsum()
        start_indices = end_indices.shift(1, fill_value=0)
        offsets = cudf.DataFrame(
            {"start_indices": start_indices, "end_indices": end_indices}
        ).to_numpy()
        return offsets

    def transform_reads_2_1d_batch(self, reads, batch_size):
        result = []
        for i in range(0, len(reads), batch_size):
            read_s = cudf.Series(reads[i : i + batch_size], name="reads")
            read_df = read_s.to_frame()
            result.append(
                read_df["reads"]
                .str.findall(".")
                .explode()
                .str.translate(self.translation_table)
                .astype("uint8")
                .to_numpy()
            )

        return np.concatenate(result).astype("uint8")

    def transform_reads_2_1d(self, reads, batch_size):
        if len(reads) > batch_size:
            return self.transform_reads_2_1d_batch(reads, batch_size)

        read_s = cudf.Series(reads, name="reads")
        read_df = read_s.to_frame()
        return (
            read_df["reads"]
            .str.findall(".")
            .explode()
            .str.translate(self.translation_table)
            .astype("uint8")
            .to_numpy()
        )

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
        reverse_comp_kmer[bpg, tbp](dev_kmers, self.kmer_length, dev_kmer_array)
        kmers = dev_kmers.copy_to_host()
        return [kmers, dev_kmer_array.copy_to_host()]

    # store the dataframe as state of this worker to be used by downstream tasks
    def calculate_kmers_multiplicity_batch(self, reads, batch_size):
        all_results = []
        for i in range(0, len(reads), batch_size):

            read_s = cudf.Series(reads[i : i + batch_size], name="reads")
            read_df = read_s.to_frame()

            replaced_df = read_df["reads"].str.replace("N", "A")
            read_df["translated"] = replaced_df.str.translate(self.translation_table)
            ngram_kmers = read_df["translated"].str.character_ngrams(
                self.kmer_length, True
            )
            exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
            numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
            result_frame = numeric_ngrams.value_counts().reset_index()
            all_results.append(result_frame)

        concat_result = (
            cudf.concat(all_results, ignore_index=True)
            .groupby("translated")
            .sum()
            .reset_index()
        )
        print(concat_result)
        concat_result.columns = ["translated", "multiplicity"]
        concat_result["multiplicity"] = concat_result["multiplicity"].clip(upper=255)
        print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        print(f"final result shape is: {concat_result.shape}")
        print(f"Kmers before calculating canonical kmers: {concat_result}")
        # [kmers_np, canonical_kmers] = self.check_rev_comp_kmer(concat_result)

        # final_kmers = (
        #     cudf.DataFrame(
        #         {"canonical": kmers_np[:, 0], "multiplicity": kmers_np[:, 1]}
        #     )
        #     .groupby("canonical")
        #     .sum()
        #     .reset_index()
        # )
        # final_kmers["multiplicity"] = final_kmers["multiplicity"].clip(upper=255)
        # print(f"Kmers after calculating canonical kmers: {final_kmers}")
        return concat_result

    # lets set arbitrary amount of batch size for canonical kmer calculation
    def calculate_kmers_multiplicity(self, reads, batch_size):

        if len(reads) > batch_size:
            return self.calculate_kmers_multiplicity_batch(reads, batch_size)

        read_s = cudf.Series(reads, name="reads")
        read_df = read_s.to_frame()

        read_df["translated"] = read_df["reads"].str.translate(self.translation_table)

        # computes canonical kmers
        ngram_kmers = read_df["translated"].str.character_ngrams(self.kmer_length, True)
        exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)

        numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
        result_frame = numeric_ngrams.value_counts().reset_index()

        result_frame.columns = ["translated", "multiplicity"]
        print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        print(f"Kmers before calculating canonical kmers: {result_frame}")
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
        print(f"Kmers after calculating canonical kmers: {final_kmers}")
        return final_kmers


# todo:refactor
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
