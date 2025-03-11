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
    backward_base,
)
from test_correction_modules import (
    one_sided_v2,
    identify_trusted_regions_v2,
)
from voting_module import cast_votes, apply_voting_result


class OneSidedTests(unittest.TestCase):

    def test_region_identification(self):

        local_read = np.array(
            [
                4,
                4,
                1,
                2,
                3,
                1,
                2,
                4,
                2,
                1,
                1,
                2,
                3,
                2,
                1,
                3,
                2,
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                5,
                2,
                1,
                3,
                2,
                1,
            ],
            dtype="uint8",
        )
        kmer_len = 13
        aux_kmer = np.zeros(kmer_len, dtype="uint8")
        seq_len = len(local_read)
        size = seq_len - kmer_len
        spectrum = []
        generate_kmers(local_read, kmer_len, spectrum)
        spectrum = count_occurence(spectrum)
        local_read[0] = 2
        local_read[12] = 4
        local_read[24] = 1
        # local_read[22] = 4
        # local_read[5] = 1
        # local_read[8] = 3
        # local_read[9] = 3

        (regions_count, regions, solids) = identify_trusted_regions_v2(
            local_read, aux_kmer, kmer_len, spectrum, seq_len, size
        )
        print(f"regions count:{regions_count} regions:{regions}")
        print(f"Solids: {solids}")

    def test_copy_kmer(self):
        local_read = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5]
        ascii_kmer = np.zeros(4, dtype="uint8")
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
