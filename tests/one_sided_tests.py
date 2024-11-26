import unittest
import numpy as np
from numpy.testing import assert_array_equal

class OneSidedTests(unittest.TestCase):
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
    def test_correction_calls(self):
        corrections_count = 0
        solids = [-1,1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,1]
        [regions_count, region_indices] = self.identify_trusted_regions(solids, 3)
        for region in range(regions_count):
            #going towards right of the region 
            #there is no next region
            if region == (regions_count - 1):
                region_end = region_indices[region][1]

                #while we are not at the end base of the read
                while region_end != (len(solids) - 1):
                    region_end += 1
                    region_indices[region][1] = region_end
                    corrections_count += 1
            #there is a next region
            if region != (regions_count - 1):
                region_end = region_indices[region][1]
                next_region_start = region_indices[region + 1][0] 

                #the loop will not stop until it does not find another region
                while region_end != (next_region_start - 1):
                   region_end += 1
                   region_indices[region][1] = region_end
                   corrections_count += 1

            #going towards left of the region

            #we are the leftmost region
            if region - 1 == -1:
                region_start = region_indices[region][0]

                #while we are not at the first base of the read
                while region_start != 0:
                  
                    region_start -= 1
                    region_indices[region][0] = region_start
                    corrections_count += 1
            #there is another region in the left side of this region 
            if region - 1 != -1:
                region_start, prev_region_end = region_indices[region][0], region_indices[region - 1][1]
                while region_start - 1 != (prev_region_end):
                    region_start -= 1
                    region_indices[region][0] = region_start
                    corrections_count += 1
        self.assertEqual(corrections_count, 6)

if __name__ == '__main__':
    unittest.main()