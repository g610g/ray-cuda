import pysam


def calculate_metrics(uncorrected_bam, corrected_bam, reference_fasta):
    """
    Calculate TP, FP, and FN by comparing uncorrected and corrected BAM files to the reference genome.

    :param uncorrected_bam: Path to the BAM file with uncorrected reads aligned to the reference.
    :param corrected_bam: Path to the BAM file with corrected reads aligned to the reference.
    :param reference_fasta: Path to the reference genome in FASTA format.
    :return: A dictionary containing TP, FP, FN counts.
    """
    # Open the BAM files
    uncorrected_aln = pysam.AlignmentFile(uncorrected_bam, "rb")
    corrected_aln = pysam.AlignmentFile(corrected_bam, "rb")

    # Open the reference genome
    reference = pysam.FastaFile(reference_fasta)

    # Store corrected reads in a dictionary for quick lookup
    corrected_reads_dict = {}
    for corrected_read in corrected_aln:
        corrected_reads_dict[corrected_read.query_name] = corrected_read

    tp = 0  # True Positives: erroneous bases that are properly corrected
    fp = 0  # False Positives: correct bases that are incorrectly corrected
    fn = 0  # False Negatives: erroneous bases that are not corrected

    # Iterate through the uncorrected reads
    for uncorrected_read in uncorrected_aln:
        # Skip unmapped reads
        if uncorrected_read.is_unmapped or uncorrected_read.reference_name is None:
            continue

        # Get the corresponding corrected read
        corrected_read = corrected_reads_dict.get(uncorrected_read.query_name)
        if (
            not corrected_read
            or corrected_read.is_unmapped
            or corrected_read.reference_name is None
        ):
            continue  # Skip if no corresponding corrected read is found or if it's unmapped

        # Ensure the reference names match
        if uncorrected_read.reference_name != corrected_read.reference_name:
            continue

        # Get the reference sequence for the read's alignment position
        try:
            ref_seq = reference.fetch(
                reference=uncorrected_read.reference_name,
                start=uncorrected_read.reference_start,
                end=uncorrected_read.reference_end,
            )
        except ValueError as e:
            print(f"Skipping read {uncorrected_read.query_name} due to error: {e}")
            continue

        # Get the read sequences
        uncorrected_seq = uncorrected_read.query_sequence
        corrected_seq = corrected_read.query_sequence

        # Ensure the read and reference sequences are the same length
        if len(uncorrected_seq) != len(ref_seq) or len(corrected_seq) != len(ref_seq):
            continue  # Skip reads with length mismatches

        # Compare each base
        for i in range(len(uncorrected_seq)):
            uncorrected_base = uncorrected_seq[i]
            corrected_base = corrected_seq[i]
            ref_base = ref_seq[i]

            if (
                uncorrected_base != ref_base
            ):  # The base was erroneous in the uncorrected read
                if corrected_base == ref_base:  # The error was corrected
                    tp += 1
                else:  # The error was not corrected
                    fn += 1
            else:  # The base was correct in the uncorrected read
                if (
                    corrected_base != ref_base
                ):  # The correct base was incorrectly corrected
                    fp += 1

            print(f"True Positives: {tp} False Positives: {fp} False Negatives: {fn}")
    # Close the files
    uncorrected_aln.close()
    corrected_aln.close()
    reference.close()

    return {"True Positives": tp, "False Positives": fp, "False Negatives": fn}


# GCF_000005845.2_ASM584v2_genomic.fna
# ERR022075_1_aligned_sorted.bam
# ERR022075_1_corrected_aligned_sorted.bam
# ecoli_30x_3perc_single_uncorrected_sorted.bam
# ecoli_70x_3perc_single_uncorrected_sorted.bam
# work-please_aligned_sorted.bam
# please_aligned_sorted.bam
# ERR022075_1.fastq
# please.fastq
# genetic-assets/final_data/ERR022075_1_corrected_sorted.bam
if __name__ == "__main__":
    original_fastq = "genetic-assets/final_data/ERR022075_1_aligned_sorted.bam"  # Path to original FASTQ file
    corrected_fastq = "genetic-assets/final_data/ERR022075_1_musket_aligned_sorted.bam"  # Path to corrected FASTQ file
    reference_fasta = "genetic-assets/final_data/GCF_000005845.2_ASM584v2_genomic.fna"  # Path to reference genome

    tp, fp, fn, tn = calculate_metrics(original_fastq, corrected_fastq, reference_fasta)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
