import pysam


def calculate_metrics(reference_fasta, original_bam, corrected_bam):
    # Open reference genome
    reference = pysam.FastaFile(reference_fasta)

    # Open original and corrected BAM files
    original_reads = pysam.AlignmentFile(original_bam, "rb")
    corrected_reads = pysam.AlignmentFile(corrected_bam, "rb")

    tp, fp, fn = 0, 0, 0

    # Iterate over each original and corrected read pair
    for original, corrected in zip(original_reads.fetch(), corrected_reads.fetch()):

        # gets the absolute positions of the reads on reference genome
        original_positions = set(original.get_reference_positions())
        corrected_positions = set(corrected.get_reference_positions())

        # gets the substring of the reference genome on where both read belongs
        ref_seq = reference.fetch(
            original.reference_name, original.reference_start, original.reference_end
        )

        # Compute TP, FP, TN, FN
        for pos in original_positions.intersection(corrected_positions):
            # only checking the pair of reads that are of same length
            # if pos in original_positions and pos in corrected_positions and (original.reference_end - original.reference_start) == (corrected.reference_end - corrected.reference_start):
            if pos in original_positions and pos in corrected_positions:
                # the relative indices for the reads
                original_index = pos - original.reference_start
                corrected_index = pos - corrected.reference_start

                if (
                    (0 <= original_index < len(original.query_sequence))
                    and (0 <= original_index < len(ref_seq))
                    and (0 <= corrected_index < len(corrected.query_sequence))
                    and (0 <= corrected_index < len(ref_seq))
                ):

                    # if there is a mismatch
                    original_is_error = (
                        original.query_sequence[original_index]
                        != ref_seq[original_index]
                    )

                    # Track if the corrected read fixed the error compared to the reference
                    corrected_is_correct = (
                        corrected.query_sequence[corrected_index]
                        == ref_seq[original_index]
                    )

                    if original_is_error and corrected_is_correct:
                        tp += 1
                    elif original_is_error and not corrected_is_correct:
                        fn += 1  # True Negative: no error originally, remains correct
                    elif not original_is_error and not corrected_is_correct:
                        fp += 1  # False Positive: was correct, incorrectly changed

        print(f"True Positive:{tp}, False Positive:{fp}, False Negative: {fn}")

    # Close files
    original_reads.close()
    corrected_reads.close()

    return {"TP": tp, "FP": fp, "FN": fn}


# Example usage
# ERR022075_1_aligned_sorted.bam
# ERR022075_1_corrected_aligned_sorted.bam
# ecoli_30x_3perc_single_uncorrected_sorted.bam
# ecoli_70x_3perc_single_uncorrected_sorted.bam
# work-please_aligned_sorted.bam
reference = "genetic-assets/GCF_000005845.2_ASM584v2_genomic.fna"
original = "genetic-assets/ERR022075_1_aligned_sorted.bam"
corrected = "genetic-assets/work-please_aligned_sorted.bam"

metrics = calculate_metrics(reference, original, corrected)
print("Metrics:", metrics)
