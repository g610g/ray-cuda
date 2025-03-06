import pysam

def calculate_fp(bam_file):
    # Open the BAM file
    bam = pysam.AlignmentFile(bam_file, "rb")

    fp_count = 0  # Counter for False Positives

    for read in bam:
        if not read.is_unmapped:  # Consider only mapped reads
            # Check for mismatches or indels
            if read.has_tag("NM"):  # NM tag stores the number of mismatches
                nm = read.get_tag("NM")
                if nm > 0:  # If there are mismatches or indels, it's a False Positive
                    fp_count += 1

    bam.close()
    return fp_count

if __name__ == "__main__":
    bam_file = "genetic-assets/ERR022075_1_corrected_aligned_sorted.bam"  # Path to your sorted BAM file
    fp = calculate_fp(bam_file)
    print(f"False Positives (FP): {fp}")
