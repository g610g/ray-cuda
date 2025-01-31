import unittest
import numpy as np
from numpy import random
from numpy.testing import assert_array_equal
from test_modules import (
    generate_kmers,
    in_spectrum,
    transform_to_key,
    count_occurence
)
from voting_module import apply_vm_result, invoke_voting


class VotingTest(unittest.TestCase):
    def test_voting_refinement(self):
        MAXLEN = 100
        kmer_len = 13
        spectrum = []

        #[MAXLEN][4]
        voting_matrix = np.zeros((MAXLEN, 4), dtype="uint16")
        # original_read = random.randint(1, 4, 100, dtype="uint8")
        original_read = [1, 2, 2, 1, 2, 1, 3, 2, 1, 2, 3, 1, 1, 2, 3, 4, 1, 2, 4, 4]
        #local read act as the copied array from original read
        local_read = original_read.copy()
        start, end = 0, len(local_read)
        size = (end - start) - kmer_len
        bases = np.zeros(4, dtype='uint8')
        generate_kmers(local_read, kmer_len, spectrum)
        local_read[3] = 3
        generate_kmers(local_read, kmer_len, spectrum)
        spectrum = count_occurence(spectrum)

        local_read[3] = 4
       
        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1
        
        max_vote = self.cast_votes(local_read, voting_matrix, end - start, kmer_len, bases, size, spectrum)
        self.apply_voting_result(local_read, voting_matrix, end - start, bases, max_vote)
        print(voting_matrix)
        print(f"Max Votes: {max_vote}")
        print(f"Read after applying the voting result: {local_read}")
        
        assert_array_equal(original_read, local_read)
        

    def cast_votes(self, local_read, vm, seq_len, kmer_len, bases, size, kmer_spectrum):
        #reset voting matrix
        for i in range(seq_len):
            for j in range(len(bases)):
                vm[i][j] = 0
        max_vote = 0
        #check each kmer within the read (planning to put the checking of max vote within this iteration)
        for ipos in range(0, size + 1):
            
            ascii_kmer = local_read[ipos: ipos + kmer_len]
            kmer = transform_to_key(ascii_kmer, kmer_len)
            if  in_spectrum(kmer_spectrum, kmer):
                continue
            for base_idx in range(kmer_len):
                original_base = ascii_kmer[base_idx]
                for base in bases:
                    # if original_base == base:
                    #     continue
                    ascii_kmer[base_idx] = base
                    candidate = transform_to_key(ascii_kmer, kmer_len)
                    if in_spectrum(kmer_spectrum, candidate):
                        vm[ipos + base_idx][base - 1] += 1
                ascii_kmer[base_idx] = original_base

        #find maximum vote
        for ipos in range(0, kmer_len):
            for idx in range(len(bases)):
                if vm[ipos][idx] >= max_vote:
                    max_vote = vm[ipos][idx]

        return max_vote
    def apply_voting_result(self, local_read, vm, seq_len, bases, max_vote):
        for ipos in range(seq_len):
            alternative_base = -1
            for base_idx in range(len(bases)):
                
                if vm[ipos][base_idx] == max_vote:
        
                    #if more than one base has the same number of votes for the same position. Correction is neglected due to ambiguity
                    if alternative_base == -1:
                        alternative_base = base_idx + 1
                        print(f"alternative base for position:{ipos} is {alternative_base}")
                    else:
                        print(f"alternative base for position:{ipos} is ambigious, bases: [{alternative_base}, {base_idx + 1}]")
                        alternative_base = -1
            #apply the base correction if we have found an alternative base
            if alternative_base >= 1:
                local_read[ipos] = alternative_base
if __name__ == "__main__":
    unittest.main()
