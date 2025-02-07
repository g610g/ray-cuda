import unittest
import numpy as np
from numpy import random 
from numpy.testing import assert_array_equal
from test_modules import (
    copy_kmer,
    count_occurence,
    generate_and_return_kmers,
    generate_kmers,
    forward_base,
    backward_base
)
from test_correction_modules import (
    one_sided_v2,
)
from voting_module import cast_votes, apply_voting_result


class OneSidedTests(unittest.TestCase):
    def test_one_sided_v2(self):
        MAX_LEN = 300

        local_read = np.random.randint(1, 5, size=100)
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
        aux_corrections = np.zeros(seq_len, dtype='uint8')
        aux_local_read = np.zeros(seq_len, dtype='uint8')
        size = seq_len - kmer_length
        min_vote = 2
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        # local_read[0] = 2
        # local_read[1] = 3
        local_read[60] = 3
        local_read[61] = 2
        local_read[59] = 4
        local_read[20] = 3
        kmers_generated = generate_and_return_kmers(local_read, kmer_length , size)
        spectrum += kmers_generated
        spectrum = count_occurence(spectrum)

        print(local_read)
        # local_read[0] = 3
        # local_read[1] = 2
        local_read[60] = 2
        local_read[61] = 1
        local_read[59] = 2
        local_read[20] = 1
        for nerr in range(1, num_errors + 1):
            for _ in range(max_iters):
                for idx in range(len(local_read)):
                    solids[idx] = -1
                copy_kmer(aux_local_read, local_read, 0, seq_len)
                corrections_count = one_sided_v2(local_read,aux_corrections, aux_kmer, aux_kmer2,  kmer_length,len(local_read), spectrum, solids, bases, nerr, num_errors - nerr + 1)

                if corrections_count > 0:
                    for idx in range(seq_len):
                        if aux_corrections[idx] != 0:
                            print(f"Changing position: {idx} to base: {aux_corrections[idx]}")
                            local_read[idx] = aux_corrections[idx]

            (max_votes, max_vote_idx)= cast_votes(local_read, votes, seq_len, kmer_length, bases, size, spectrum, aux_kmer)
            if max_votes == 0:
                print("max votes is zero")
            elif max_votes >= min_vote:
                print(f"max votes is {max_votes} in index {max_vote_idx}")
                apply_voting_result(local_read, votes, seq_len, bases, max_votes)

        assert_array_equal(local_read, correct_read)
    # def test_forward_base(self):
    #     local_read = np.array([4, 4, 1, 2, 3, 1, 2, 4, 2, 1, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 1, 3, 2, 1], dtype='uint8')
    #     kmer_length = 13
    #     aux_kmer = np.zeros(kmer_length, dtype='uint8')
    #     copy_kmer(aux_kmer, local_read, 0, 13)
    #     forward_base(aux_kmer, local_read[13], kmer_length)
    #     assert_array_equal(aux_kmer, local_read[1: 14])
    # def test_backward_base(self):
    #     local_read = np.array([4, 4, 1, 2, 3, 1, 2, 4, 2, 1, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 1, 3, 2, 1], dtype='uint8')
    #     kmer_length = 13
    #     aux_kmer = np.zeros(kmer_length, dtype='uint8')
    #     copy_kmer(aux_kmer, local_read, 1, 14)
    #     backward_base(aux_kmer, 1, kmer_length)
    #     answer = local_read[0: 13]
    #     answer[0] = 1
    #     assert_array_equal(aux_kmer, answer)
    def test_region_identification(self):

        local_read = np.array([4, 4, 1, 2, 3, 1, 2, 4, 2, 1, 1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 1, 3, 2, 1], dtype='uint8')
        # local_read = np.array([1, 2, 2, 3, 1, 2, 1, 1, 2, 1], dtype='uint8')
        kmer_len = 13
        aux_kmer = np.zeros(kmer_len, dtype='uint8')
        seq_len = len(local_read)
        size = seq_len - kmer_len
        spectrum = []
        generate_kmers(local_read, kmer_len, spectrum)
        spectrum = count_occurence(spectrum)
        local_read[1] = 2
        local_read[35] = 4
        local_read[34] = 4
        local_read[22] = 4
        # local_read[5] = 1
        # local_read[8] = 3
        # local_read[9] = 3

        # (regions_count, regions, solids)  = identify_trusted_regions_v2(local_read, aux_kmer, kmer_len, spectrum, seq_len, size)
        # print(f"regions count:{regions_count} regions:{regions}")
        # print(f"Solids: {solids}")
    def test_copy_kmer(self):
        local_read = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5]
        ascii_kmer = np.zeros(4, dtype='uint8')
        copy_kmer(ascii_kmer, local_read, 2, 6)
        print(ascii_kmer)

def transform_to_key(ascii_kmer, len):
    multiplier = 1
    key = 0
    while len != 0:
        key += ascii_kmer[len - 1] * multiplier
        multiplier *= 10
        len -= 1

    return key


def in_spectrum(kmer, spectrum):
    if kmer in spectrum:
        return True

    return False


if __name__ == "__main__":
    unittest.main()
