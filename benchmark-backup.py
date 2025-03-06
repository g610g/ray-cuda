import pysam
from Bio import SeqIO


def classify_reads(original_fastq, corrected_fastq, reference_fasta):
    # Load reference genome
    reference = list(SeqIO.parse(reference_fasta, "fasta"))[0].seq

    # Initialize counters
    tp = fp = fn = tn = 0

    # Open original and corrected FASTQ files
    with pysam.FastxFile(original_fastq) as original, pysam.FastxFile(
        corrected_fastq
    ) as corrected:
        for orig_read, corr_read in zip(original, corrected):
            # Check if the read is erroneous (not matching the reference)
            orig_seq = orig_read.sequence
            corr_seq = corr_read.sequence

            # Align original read to reference (simplified alignment for demonstration)
            is_erroneous = orig_seq not in reference

            # Classify based on algorithm outcome
            if orig_seq != corr_seq:  # Read was fixed or discarded
                if is_erroneous:
                    tp += 1  # True Positive
                else:
                    fp += 1  # False Positive
            else:  # Read was unchanged
                if is_erroneous:
                    fn += 1  # False Negative
                else:
                    tn += 1  # True Negative

            return {"True Positives": tp, "False Positives": fp, "False Negatives": fn}
    return tp, fp, fn, tn


if __name__ == "__main__":
    original_fastq = "original_reads.fastq"  # Path to original FASTQ file
    corrected_fastq = "corrected_reads.fastq"  # Path to corrected FASTQ file
    reference_fasta = "reference_genome.fna"  # Path to reference genome

    tp, fp, fn, tn = classify_reads(original_fastq, corrected_fastq, reference_fasta)

    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN): {tn}")
