from Bio import SeqIO
import fastq_parser

reads1 = {rec.id: rec.seq for rec in SeqIO.parse("genetic-assets/ERR022075_1_corrected.fastq", "fastq")}
reads2 = {rec.id: rec.seq for rec in SeqIO.parse("genetic-assets/please.fastq", "fastq")}

for read_id in reads1:
    if read_id in reads2:
        if reads1[read_id] != reads2[read_id]:
            print(f"Read {read_id} differs: {reads1[read_id]} vs {reads2[read_id]}")
    else:
        print(f"Read {read_id} is missing in file2")
