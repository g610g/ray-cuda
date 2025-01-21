import unittest
import numpy as np
from numpy import random
from test_modules import generate_kmers, identify_solid_bases
from test_correction_modules import correct_reads_two_sided


class TwoSidedTest(unittest.TestCase):
    def test_correcting_edge_bases(self):
        MAX_LEN = 300
        spectrum = []
        kmer_length = 13
        local_read = random.randint(1, 4, 100, dtype="uint8")

        bases = np.zeros(4, dtype="uint8")
        start, end = 0, len(local_read)
        solids = np.zeros(MAX_LEN, dtype="int8")
        # seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1
        after_correction_solids = solids.copy()
        generate_kmers(local_read, kmer_length, spectrum)

        # modify random bases in the local read
        for _idx in range(4):
            random_idx = random.randint(0, 70)
            local_read[random_idx] = random.randint(1, 4)

        identify_solid_bases(
            local_read, 0, len(local_read), kmer_length, spectrum, solids
        )

        print(solids)
        for base_idx in range(0, end - start):
            # the base needs to be corrected
            if (
                solids[base_idx] == -1
                and base_idx >= (kmer_length - 1)
                and base_idx <= (end - start) - kmer_length
            ):

                left_portion = local_read[base_idx - (kmer_length - 1) : base_idx + 1]
                right_portion = local_read[base_idx : base_idx + kmer_length]

                correct_reads_two_sided(
                    base_idx,
                    local_read,
                    kmer_length,
                    spectrum,
                    bases,
                    left_portion,
                    right_portion,
                )

            # the leftmost bases of the read
            if solids[base_idx] == -1 and base_idx < (kmer_length - 1):
                pass
            # the rightmost bases of the read
            if solids[base_idx] == -1 and base_idx > (end - start) - kmer_length:
                pass

        identify_solid_bases(
            local_read,
            0,
            len(local_read),
            kmer_length,
            spectrum,
            after_correction_solids,
        )
        print(after_correction_solids)


if __name__ == "__main__":
    unittest.main()
