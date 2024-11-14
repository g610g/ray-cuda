import unittest
import numpy as np 
from numpy.testing import assert_array_equal

#THIS IS THE TESTING MODULE WHERE I CONDUCT UNIT TESTS ON THE LOGICS THAT WAS IMPLEMENTED ON THE KERNEL LOGIC.
#tHE UNIT TESTING IS SEPERATED AND IS EXECUTED ON A CPU RUNTIME AND NOT ON gpu RUNTIME AS I FOUND IT HARD TO FIND RESOURCES IN ORDER TO UNIT TEST KERNEL CUDA CODES

class CorrectionTests(unittest.TestCase):
    def test_identifying_trusted_regions(self):

        [regions_num, _] = self.identify_trusted_regions([-1,1,1,1,1,1,1,1,1,1,1], 3) 

        self.assertEqual(regions_num, 1)

    def test_assert_index_counting_right_orientation(self):
        shared_reads = [3,4,2,4,3,2,1,2,3,5,4,4,4,4,3,4,2,4,3,2,1,2,3,5,4,4,4,4]
        kmer_len = 13
        region_end = 17
        curr_kmer = shared_reads[(region_end - (kmer_len - 1)): region_end + 1]
        forward_kmer = shared_reads[(region_end - (kmer_len - 1)) + 1: region_end + 2]

        self.assertEqual(curr_kmer , [ 2, 1, 2, 3, 5, 4, 4, 4, 4, 3, 4, 2, 4])
        self.assertEqual(forward_kmer, [1,2,3,5,4,4,4,4,3,4,2,4,3])

    def test_assert_index_counting_left_orientation(self):
        shared_reads = [3,4,2,4,3,2,1,2,3,5,4,4,4,4,3,4,2,4,3,2,1,2,3,5,4,4,4,4]
        kmer_len = 13
        region_start = 2

        curr_kmer = shared_reads[region_start: region_start + kmer_len]
        backward_kmer = shared_reads[region_start - 1: region_start + (kmer_len - 1)]

        self.assertEqual(curr_kmer, [2,4,3,2,1,2,3,5,4,4,4,4,3])
        self.assertEqual(backward_kmer, [4,2,4,3,2,1,2,3,5,4,4,4,4])

    def test_assert_end_region(self):

        shared_reads = [3,4,2,4,3,2,1,2,3,5,4,4,4,4,3,4,2,4,3,2,1,2,3,5,4,4,4,4]
        kmer_len = 13
        region_end = len(shared_reads) - 2
        curr_kmer = shared_reads[(region_end - (kmer_len - 1)): region_end + 1]
        forward_kmer = shared_reads[(region_end - (kmer_len - 1)) + 1: region_end + 2]

        self.assertEqual(curr_kmer , [ 3,4,2,4,3,2,1,2,3,5,4,4,4])
        self.assertEqual(forward_kmer, [4,2,4,3,2,1,2,3,5,4,4,4,4])

    def test_assert_transform_to_key(self):
        shared_reads = [3,4,2,4,3,2,1,2,3,5,4,4,4,4,3,4,2,4,3,2,1,2,3,5,4,4,4,4]
        kmer_len = 13
        key = self.transform_to_key(shared_reads[1:kmer_len + 1], kmer_len)
        print(f"key generated: {key}")
        self.assertEqual(key, 4243212354444)

    def test_binary_search_2d(self):
        sorted_arr = [[121231331133313, 2], [121231333123312, 2], [121231333123322, 2], [221212334123312, 2], [521212334123312, 2]] 
        needle = 221212334123312

        idx = self.binary_search_2d(sorted_arr, needle) 
        self.assertEqual(idx, 3)

    def binary_search_2d(self, sorted_arr, needle):
        sorted_arr_len = len(sorted_arr)
        right = sorted_arr_len - 1
        left = 0

        while(left <= right):
            middle  = (left + right) // 2
            if sorted_arr[middle][0] == needle:
                return middle

            elif sorted_arr[middle][0] > needle:
                right = middle - 1

            elif sorted_arr[middle][0] < needle:
                left = middle + 1
        return -1

    def transform_to_key(self, ascii_kmer, len):
        multiplier = 1
        key = 0
        while(len != 0):
            key += (ascii_kmer[len - 1] * multiplier)
            multiplier *= 10
            len -= 1

        return key

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
