import unittest
import numpy as np 
from numpy.testing import assert_array_equal

#THIS IS THE TESTING MODULE WHERE I CONDUCT UNIT TESTS ON THE LOGICS THAT WAS IMPLEMENTED ON THE KERNEL LOGIC.
#tHE UNIT TESTING IS SEPERATED AND IS EXECUTED ON A CPU RUNTIME AND NOT ON gpu RUNTIME AS I FOUND IT HARD TO FIND RESOURCES IN ORDER TO UNIT TEST KERNEL CUDA CODES

class CorrectionTests(unittest.TestCase):
    def test_identifying_trusted_regions(self):

        [regions_num, _] = self.identify_trusted_regions([-1,1,1,1,1,1,1,1,1,1,1], 3) 

        self.assertEqual(regions_num, 1)

    def test_assert_index_counting(self):
        shared_reads = [3,4,2,4,3,2,1,2,3,5,4,4,4,4,3,4,2,4,3,2,1,2,3,5,4,4,4,4]
        kmer_len = 13
        region_end = 17
        curr_kmer = shared_reads[(region_end - (kmer_len - 1)): region_end + 1]
        forward_kmer = shared_reads[(region_end - (kmer_len - 1)) + 1: region_end + 2]

        self.assertEqual(curr_kmer , [ 2, 1, 2, 3, 5, 4, 4, 4, 4, 3, 4, 2, 4])
        self.assertEqual(forward_kmer, [1,2,3,5,4,4,4,4,3,4,2,4,3])
    def test_assert_region_indices(self):
        [_, region_indices] = self.identify_trusted_regions([-1,1,1,1,1,1,1,1,1,1,1], 3) 
        expected = np.zeros((10, 2), dtype='uint8')
        expected[0][0], expected[0][1] = 1, 10
        

        assert_array_equal(region_indices, expected)

    def identify_trusted_regions(self, solids, kmer_len):
        current_indices_idx = 0
        base_count = 0
        region_start = 0
        region_end = 0
        region_indices = np.zeros((10, 2), dtype='uint8')
        
        for idx in range(len(solids)):

            if base_count >= kmer_len and solids[idx] == -1:

                region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, region_end 
                region_start = idx + 1
                region_end = idx + 1
                current_indices_idx += 1
                base_count = 0

        
            if solids[idx] == -1 and base_count < kmer_len:
                region_start = idx + 1
                region_end = idx + 1
                base_count = 0

            if solids[idx] == 1:
                region_end = idx
                base_count += 1

        if base_count >= kmer_len:

            region_indices[current_indices_idx][0], region_indices[current_indices_idx][1] = region_start, region_end
            current_indices_idx += 1

        return [current_indices_idx, region_indices]
if __name__ == '__main__':
    unittest.main()
