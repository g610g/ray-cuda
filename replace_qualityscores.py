def replace_quality(sim_fastq, qual_file, output_fastq):
    with open(sim_fastq, 'r') as sim_fq, open(qual_file, 'r') as qual_fq, open(output_fastq, 'w') as out_fq:
        while True:
            header = sim_fq.readline().strip()
            seq = sim_fq.readline().strip()
            plus = sim_fq.readline().strip()
            sim_quality = sim_fq.readline().strip()

            quality = qual_fq.readline().strip()

            if not header or not seq or not plus or not sim_quality or not quality:
                break

            out_fq.write(f"{header}\n")
            out_fq.write(f"{seq}\n")
            out_fq.write(f"{plus}\n")
            out_fq.write(f"{quality}\n")

# Replace quality scores for single-end reads
replace_quality("genetic-assets/final_data/ecoli_datasets/ecoli_30coverage_1perc.bwa.read1.fastq", "genetic-assets/final_data/ecoli_datasets/ecoli_qualityscores.txt", "genetic-assets/final_data/ecoli_datasets/ecoli_30coverage_1perc_modified_quality.read1.fastq")
