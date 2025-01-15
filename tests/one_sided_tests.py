import unittest
import numpy as np
from numpy.testing import assert_equal, assert_array_equal
from test_modules import identify_trusted_regions
from test_correction_modules import correct_read_one_sided_right,correct_read_one_sided_left


class OneSidedTests(unittest.TestCase):
    def test_one_sided_core(self):
        MAX_LEN = 300
        spectrum = []
        kmer_length = 13
        local_read = [1, 3, 2, 1, 2, 4, 1, 2, 3, 5, 4, 3, 4, 2, 1, 3, 5, 3, 2, 3]
        alternatives = np.zeros((4, 2), dtype="uint32")
        num_kmers  = len(local_read) - (kmer_length - 1)
        correction_tracker = np.zeros(MAX_LEN, dtype="uint8")
        bases = np.zeros(4, dtype="uint8") 
        start, end = 0, len(local_read)
        region_indices = np.zeros((10, 2), dtype="int8")
        solids = np.zeros(MAX_LEN, dtype="int8")

        #seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        #seeding bases
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        max_idx = len(local_read) - 1
        #modify local read to simulate read with error bases
        #trying to put error within ends of the read
        local_read[0], local_read[1], local_read[2] = 2, 2, 1
        local_read[0], local_read[1], local_read[2] = 2, 2, 1
        local_read[max_idx], local_read[max_idx - 1], local_read[max_idx - 2] = 2, 1, 2

        regions_count = identify_trusted_regions(0, len(local_read), spectrum, local_read, kmer_length, region_indices, solids)

        if regions_count == 0:
                return
            # no unit tests for this part yet
        for region in range(regions_count):
            # going towards right of the region

            # there is no next region
            if region == (regions_count - 1):
                region_end = region_indices[region][1]

                # while we are not at the end base of the read
                while region_end != (end - start) - 1:
                    if not correct_read_one_sided_right(
                        local_read,
                        region_end,
                        spectrum,
                        kmer_length,
                        bases,
                        alternatives,
                        correction_tracker,
                        num_kmers - 1,
                        end - start,
                    ):
                        break

                    # extend the portion of region end for successful correction
                    else:
                        region_end += 1
                        region_indices[region][1] = region_end

            # there is a next region
            if region != (regions_count - 1):
                region_end = region_indices[region][1]
                next_region_start = region_indices[region + 1][0]

                # the loop will not stop until it does not find another region
                while region_end != (next_region_start - 1):
                    if not correct_read_one_sided_right(
                        local_read,
                        region_end,
                        spectrum,
                        kmer_length,
                        bases,
                        alternatives,
                        correction_tracker,
                        num_kmers - 1,
                        end - start,
                    ):
                        # fails to correct this region and on this orientation
                        break

                    # extend the portion of region end for successful correction
                    else:
                        region_end += 1
                        region_indices[region][1] = region_end

            # going towards left of the region
            # we are the leftmost region
            if region - 1 == -1:
                region_start = region_indices[region][0]

                # while we are not at the first base of the read
                while region_start != 0:
                    if not correct_read_one_sided_left(
                        local_read,
                        region_start,
                        spectrum,
                        kmer_length,
                        bases,
                        alternatives,
                        correction_tracker,
                        num_kmers - 1,
                        end - start,
                    ):
                        break
                    else:
                        region_start -= 1
                        region_indices[region][0] = region_start

            # there is another region in the left side of this region
            if region - 1 != -1:
                region_start, prev_region_end = (
                    region_indices[region][0],
                    region_indices[region - 1][1],
                )
                while region_start - 1 != (prev_region_end):

                    if not correct_read_one_sided_left(
                        local_read,
                        region_start,
                        spectrum,
                        kmer_length,
                        bases,
                        alternatives,
                        correction_tracker,
                        num_kmers - 1,
                        end - start,
                    ):
                        break
                    else:
                        region_start -= 1
                        region_indices[region][0] = region_start


        print(solids)
        print(region_indices)

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
    
    def test_kmer_correction_counter_ends(self):
        kmer_len = 3
        reads = np.arange(11)
        kmer_counter_list = np.zeros((len(reads) - (kmer_len - 1)), dtype='uint8')
        self.mark_kmer_counter(0, kmer_counter_list, kmer_len, 8, len(reads))

        assert_array_equal(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),kmer_counter_list) 

    #testing for 3mers
    def test_onesided_lookahead(self):
        spectrum = []
        kmer_length = 13
        local_read = [1, 3, 2, 1, 2, 4, 1, 2, 3, 5, 4, 3, 4, 2, 1, 3, 5, 3, 2, 3]

        #add in spectrum
        for idx in range(0, len(local_read) - (kmer_length - 1)):
            spectrum.append(self.transform_to_key(local_read[idx: idx + kmer_length], kmer_length))

        max_idx = len(local_read) - 1
        local_read[max_idx] = 4
        res = self.lookahead_validation(kmer_length, local_read, spectrum, max_idx, 3)
        assert_equal(True, res)

    def lookahead_validation(
        self,
        kmer_length,
        local_read,
        kmer_spectrum,
        modified_base_idx,
        alternative_base,
        neighbors_max_count=2,
    ):
        # this is for base that has kmers that covers < neighbors_max_count
        if modified_base_idx < neighbors_max_count:
            num_possible_neighbors = modified_base_idx + 1
            counter = modified_base_idx
            min_idx = 0
            for _ in range(num_possible_neighbors):
                alternative_kmer = local_read[min_idx: min_idx+kmer_length]
                alternative_kmer[counter] = alternative_base
                transformed_alternative_kmer = self.transform_to_key(alternative_kmer, kmer_length)
                print(f"min index: {min_idx}, max idx: {min_idx + kmer_length} counter: {counter} alternative kmer: {transformed_alternative_kmer}")
                if not in_spectrum( transformed_alternative_kmer, kmer_spectrum):
                    return False
                min_idx += 1
                counter -= 1
            return True

        # for bases that are modified outside the "easy range"
        if modified_base_idx >= len(local_read) - neighbors_max_count:
            num_possible_neighbors = (len(local_read) - 1) - modified_base_idx
            min_idx = modified_base_idx - (kmer_length - 1)
            max_idx = modified_base_idx
            counter = kmer_length - 1

            for _ in range(num_possible_neighbors + 1):
                alternative_kmer = local_read[min_idx: min_idx + kmer_length]
                alternative_kmer[counter] = alternative_base
                transformed_alternative_kmer = self.transform_to_key(alternative_kmer, kmer_length)
                # print(f"min index: {min_idx}, max idx: {min_idx + kmer_length} counter: {counter} alternative kmer: {transformed_alternative_kmer}")

                if not in_spectrum(transformed_alternative_kmer, kmer_spectrum):
                    return False
                min_idx += 1
                counter -= 1
            return True

        if modified_base_idx < (kmer_length - 1):
            min_idx = 0
            max_idx = kmer_length
            counter = modified_base_idx
        else:
            # this is the modified base idx that are within the range of "easy range"
            min_idx = modified_base_idx - (kmer_length - 1)
            max_idx = modified_base_idx
            counter = kmer_length - 1

        for _idx in range(neighbors_max_count):
            if min_idx > max_idx:
                return False
            alternative_kmer = local_read[min_idx : min_idx + kmer_length]
            print(f"alternative kmer: {alternative_kmer}")
            print(f"min index: {min_idx}, max idx: {min_idx + kmer_length} counter: {counter}")
            alternative_kmer[counter] = alternative_base
            transformed_alternative_kmer = self.transform_to_key(alternative_kmer, kmer_length)
            if not in_spectrum(transformed_alternative_kmer, kmer_spectrum):
                return False

            min_idx += 1
            counter -= 1

        # returned True meaning the alternative base which sequencing error occurs is (valid)?
        return True

    def mark_kmer_counter(self, base_idx, kmer_counter_list, kmer_len, max_kmer_idx, read_length):
        if base_idx < (kmer_len - 1):
            for idx in range(0, base_idx + 1):
                kmer_counter_list[idx] += 1
            return

        if base_idx > (read_length - (kmer_len - 1)):
            min = base_idx - (kmer_len - 1)
            for idx in range(min, max_kmer_idx + 1):
                kmer_counter_list[idx] += 1
            return

        min = base_idx - (kmer_len - 1)
        if base_idx > max_kmer_idx:
            for idx in range(min, max_kmer_idx + 1):
                kmer_counter_list[idx] += 1
            return
        for idx in range(min, base_idx + 1):
            kmer_counter_list[idx] += 1
        return

    def transform_to_key(self, ascii_kmer, len):
        multiplier = 1
        key = 0
        while(len != 0):
            key += (ascii_kmer[len - 1] * multiplier)
            multiplier *= 10
            len -= 1

        return key

def transform_to_key( ascii_kmer, len):
    multiplier = 1
    key = 0
    while(len != 0):
        key += (ascii_kmer[len - 1] * multiplier)
        multiplier *= 10
        len -= 1

    return key
def in_spectrum(kmer, spectrum):
    if kmer in spectrum:
        return True

    return False
def generate_kmers(read, kmer_length, kmer_spectrum):
    for idx in range(0, len(read) - (kmer_length - 1)):
        kmer_spectrum.append(transform_to_key(read[idx: idx + kmer_length], kmer_length))

if __name__ == '__main__':
    unittest.main()
