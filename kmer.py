import ray
import math
import numpy as np
import cudf


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
        read_df = cudf.DataFrame({"reads": reads})
        str_lens = read_df["reads"].str.len()
        end_indices = str_lens.cumsum()
        start_indices = end_indices.shift(1, fill_value=0)
        offsets = cudf.DataFrame(
            {"start_indices": start_indices, "end_indices": end_indices}
        ).to_numpy()
        return offsets

    def transform_reads_2_1d_batch(self, reads, batch_size):
        result = np.array([])
        for i in range(0, len(reads), batch_size):
            read_df = cudf.DataFrame({"reads": reads[i : i + batch_size]})
            result = np.append(
                result,
                read_df["reads"]
                .str.findall(".")
                .explode()
                .str.translate(self.translation_table)
                .astype("uint8")
                .to_numpy()
            )

        return result

    def transform_reads_2_1d(self, reads, batch_size):
        read_df = cudf.DataFrame({"reads": reads})
        if len(reads) > batch_size :
            return self.transform_reads_2_1d_batch(reads, batch_size) 
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

    def calculate_kmers_multiplicity_batch(self, reads, batch_size):
        all_results = []
        for i in range(0, len(reads), batch_size):
            read_s = cudf.Series(reads[i : i + batch_size], name="reads")
            read_df = read_s.to_frame()
            read_df["translated"] = read_df["reads"].str.translate(
                self.translation_table
            )
            ngram_kmers = read_df["translated"].str.character_ngrams(
                self.kmer_length, True
            )
            exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
            numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
            result_frame = numeric_ngrams.value_counts().reset_index()
            all_results.append(result_frame)

        final_result = (
            cudf.concat(all_results).groupby("translated").sum().reset_index()
        )
        final_result.columns = ["translated", "multiplicity"]
        print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        print(f"final result shape is: {final_result.shape}")
        print(f"final result is: {final_result}")
        return final_result

    def calculate_kmers_multiplicity(self, reads, batch_size):

        if len(reads) > batch_size:
            return self.calculate_kmers_multiplicity_batch(reads, batch_size)

        read_s = cudf.Series(reads, name="reads")
        read_df = read_s.to_frame()

        read_df["translated"] = read_df["reads"].str.translate(self.translation_table)

        ngram_kmers = read_df["translated"].str.character_ngrams(self.kmer_length, True)

        exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
        numeric_ngrams = exploded_ngrams.astype("uint64").reset_index(drop=True)
        result_frame = numeric_ngrams.value_counts().reset_index()

        result_frame.columns = ["translated", "multiplicity"]

        print(f"used kmer len for extracting kmers is: {self.kmer_length}")
        return result_frame

    def give_lengths_of_kmers(self, reads):
        read_s = cudf.Series(reads, name="reads")

        read_df = read_s.to_frame()

        read_df["translated"] = read_df["reads"].str.translate(self.translation_table)
        ngram_kmers = read_df["translated"].str.character_ngrams(self.kmer_length, True)

        exploded_ngrams = ngram_kmers.explode().reset_index(drop=True)
        kmer_lens = exploded_ngrams.str.len().reset_index(drop=True)
        kmer_lens_df = kmer_lens.to_frame()
        kmer_lens_df.columns = ["lengths"]

        return kmer_lens_df


# todo:refactor
def calculatecutoff_threshold(occurence_data, bin):

    hist_vals, bin_edges = np.histogram(occurence_data, bins=bin)

    print((hist_vals[:30]))
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    print(bin_centers[:30])

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
