import unittest
import numpy as np
from numpy import random
from numpy.testing import assert_array_equal
from test_modules import (
    generate_kmers,
    in_spectrum,
    transform_to_key,
    identify_solid_bases,
)
from voting_module import apply_vm_result, invoke_voting


class VotingTest(unittest.TestCase):
    def test_voting_indices(self):
        pass
    def test_voting_refinement(self):
        MAXLEN = 100
        kmer_len = 13
        spectrum = []
        voting_matrix = np.zeros((4, MAXLEN), dtype="uint16")
        original_read = random.randint(1, 4, 100, dtype="uint8")
        
        #local read act as the copied array from original read
        local_read = original_read.copy()

        start, end = 0, len(local_read)
        bases = np.zeros(4, dtype='uint8')
        generate_kmers(local_read, kmer_len, spectrum)
        print(f"Correct read: {local_read}")

        solids = np.zeros(MAXLEN, dtype="int8")

        # seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1
        # modifying random bases to be error
        for _ in range(10):
            random_idx = random.randint(48, 70)
            local_read[random_idx] = random.randint(1, 4)
        identify_solid_bases(local_read, start, end, kmer_len, spectrum, solids)

        print(f"Solids array: {solids}")

        for idx in range(start, end - (kmer_len - 1)):
            curr_kmer = transform_to_key(local_read[idx : idx + kmer_len], kmer_len)

            # invoke voting if the kmer is not in spectrum
            if not in_spectrum(spectrum, curr_kmer):
                invoke_voting(
                    voting_matrix, spectrum, bases, kmer_len, idx, local_read, start
                )

        print(voting_matrix)
        # apply the result of the vm into the reads
        apply_vm_result(voting_matrix, local_read, start, end)
        print(f"Read after applying the voting result: {local_read}")
        
        assert_array_equal(local_read, original_read)

if __name__ == "__main__":
    unittest.main()
