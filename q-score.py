
import sys
from Bio import SeqIO, Seq
from Bio.Seq import MutableSeq
from itertools import islice

def batch_dez_nuts(iterator, batch_size):
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

sequences = SeqIO.parse(sys.argv[1],'fastq') 
my_seq = Seq.Seq("ACGTGGCCACA")


batches = batch_dez_nuts(sequences, 200)
for batch in batches:
    print(len(batch))
# for batch in batches:
#     for seq in batch:
#         print(seq)
# for read_sequence in sequences:
#
#     mutable_seq = MutableSeq(read_sequence.seq)
#     print(type(mutable_seq))
#     replaced_seq = mutable_seq.replace('A', 'N')
#     read_sequence.seq = Seq.Seq(replaced_seq)
#     #
#     print(read_sequence.seq)
    # print(read_sequence.letter_annotations['phred_quality'])


