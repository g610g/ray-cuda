import unittest
import numpy as np
from numpy import random 
from numpy.testing import assert_array_equal
from test_modules import (
    count_occurence,
    generate_and_return_kmers,
    generate_kmers,
)
from test_correction_modules import (
    one_sided_v2_host,
)
from voting_module import cast_votes, apply_voting_result


class OneSidedTests(unittest.TestCase):
    def test_one_sided_v2(self):
        MAX_LEN = 300

        local_read = np.array([4, 4, 1, 2, 3, 1, 2, 4, 2, 1, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 4, 1, 2, 3, 4, 2, 2, 1, 3, 2, 1], dtype='uint8')
        correct_read = local_read.copy()
        kmer_length = 13
        aux_kmer = np.zeros(kmer_length, dtype='uint8')
        aux_kmer2 = np.zeros(kmer_length, dtype='uint8')
        votes = np.zeros((MAX_LEN, 4), dtype='uint8')
        solids = np.zeros(300, dtype='int8')
        bases = np.zeros(4, dtype='uint8')
        spectrum = []
        num_errors = 4
        max_iters = 2
        seq_len = len(local_read)
        original_read = np.zeros(seq_len, dtype='uint8')
        size = seq_len - kmer_length
        min_vote = 2
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        # local_read[0] = 2
        # local_read[1] = 3
        # local_read[35] = 3
        # local_read[34] = 3
        # local_read[22] = 3
        local_read[2] = 3
        kmers_generated = generate_and_return_kmers(local_read, kmer_length , size)
        spectrum += kmers_generated
        spectrum = count_occurence(spectrum)

        print(spectrum)
        # local_read[0] = 3
        # local_read[1] = 2
        # local_read[35] = 4
        local_read[2] = 2
        # local_read[34] = 4
        # local_read[22] = 4
        for nerr in range(1, num_errors + 1):
            for _ in range(max_iters):
                for idx in range(seq_len):
                    solids[idx] = -1
                original_read = local_read.copy()

                one_sided_v2_host(local_read, original_read, aux_kmer, aux_kmer2,  kmer_length, len(local_read), spectrum, solids, bases, nerr, num_errors - nerr + 1)

                print(original_read)
                local_read = original_read.copy()

            # (max_votes, max_vote_idx)= cast_votes(local_read, votes, seq_len, kmer_length, bases, size, spectrum, aux_kmer)
            # if max_votes == 0:
            #     print("max votes is zero")
            # elif max_votes >= min_vote:
            #     print(f"max votes is {max_votes} in index {max_vote_idx}")
            #     apply_voting_result(local_read, votes, seq_len, bases, max_votes)

        assert_array_equal(local_read, correct_read)

if __name__ == "__main__":
    unittest.main()
