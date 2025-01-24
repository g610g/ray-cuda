import unittest
import numpy as np
from numpy import random
from numpy.testing import assert_array_equal
from test_modules import identify_trusted_regions, generate_kmers
from test_correction_modules import (
    correct_read_one_sided_right,
    correct_read_one_sided_left,
)


class OneSidedTests(unittest.TestCase):
    def test_one_sided_core(self):
        MAX_LEN = 300
        spectrum = []
        kmer_length = 13
        # local_read = random.randint(1, 4, 100, dtype="uint8")
        local_read = [4, 2, 1, 2, 4, 1, 2, 3, 4, 2, 1, 5, 2, 3, 4, 4, 1, 2, 3, 4]
        # array used to store base alternative and its corresponding kmer count
        alternatives = np.zeros((4, 2), dtype="uint32")
        local_read_len = len(local_read)
        num_kmers = len(local_read) - (kmer_length - 1)
        # array that keep tracks of corrections made for every kmer
        correction_tracker = np.zeros(MAX_LEN, dtype="uint8")
        bases = np.zeros(4, dtype="uint8")
        start, end = 0, len(local_read)
        region_indices = np.zeros((10, 2), dtype="int8")
        solids = np.zeros(MAX_LEN, dtype="int8")

        print(local_read)
        # seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        spectrum.append(32124)
        spectrum.append(11241)
        print(spectrum)
        local_read[0] = 1
        local_read[1] = 3
        # modify random bases in the local read
        # for _ in range(10):
        #     random_idx = random.randint(0, local_read_len)
        #     local_read[random_idx] = random.randint(1, 4)

        regions_count = identify_trusted_regions(
            0,
            len(local_read),
            spectrum,
            local_read,
            kmer_length,
            region_indices,
            solids,
        )

        print(solids)
        print(region_indices)
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
                        print(
                            f"Correction toward right for idx: {region_end + 1} is not successful"
                        )
                        break

                    # extend the portion of region end for successful correction
                    else:
                        print(
                            f"Correction in index {region_end + 1} orientation going right is succesful"
                        )
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
                        print(
                            f"Correction toward right for idx: {region_end + 1} is not successful "
                        )
                        break

                    # extend the portion of region end for successful correction
                    else:
                        print(
                            f"Correction in index {region_end + 1} orientation going right is succesful"
                        )
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
                        print(
                            f"Correction toward left for idx: {region_start - 1} is not successful "
                        )
                        break
                    else:
                        print(
                            f"Correction in index {region_start - 1} orientation going left is succesful"
                        )
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
                        print(
                            f"Correction toward left for idx: {region_start - 1} is not successful "
                        )
                        break
                    else:
                        print(
                            f"Correction in index {region_start - 1} orientation going left is successful"
                        )
                        region_start -= 1
                        region_indices[region][0] = region_start

        # reinitialize solids array
        for idx in range(len(local_read)):
            solids[idx] = -1

        regions_count = identify_trusted_regions(
            0,
            len(local_read),
            spectrum,
            local_read,
            kmer_length,
            region_indices,
            solids,
        )

        print("Result after correction")

        print(solids)
        print(local_read)

    def identify_trusted_regions(self, solids, kmer_len):
        current_indices_idx = 0
        base_count = 0
        region_start = 0
        region_end = 0
        region_indices = np.zeros((10, 2), dtype="uint8")

        for idx in range(len(solids)):

            if base_count >= kmer_len and solids[idx] == -1:

                (
                    region_indices[current_indices_idx][0],
                    region_indices[current_indices_idx][1],
                ) = (region_start, region_end)
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

            (
                region_indices[current_indices_idx][0],
                region_indices[current_indices_idx][1],
            ) = (region_start, region_end)
            current_indices_idx += 1

        return [current_indices_idx, region_indices]

    def test_correction_calls(self):
        corrections_count = 0
        solids = [-1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1]
        [regions_count, region_indices] = self.identify_trusted_regions(solids, 3)
        for region in range(regions_count):
            # going towards right of the region
            # there is no next region
            if region == (regions_count - 1):
                region_end = region_indices[region][1]

                # while we are not at the end base of the read
                while region_end != (len(solids) - 1):
                    region_end += 1
                    region_indices[region][1] = region_end
                    corrections_count += 1
            # there is a next region
            if region != (regions_count - 1):
                region_end = region_indices[region][1]
                next_region_start = region_indices[region + 1][0]

                # the loop will not stop until it does not find another region
                while region_end != (next_region_start - 1):
                    region_end += 1
                    region_indices[region][1] = region_end
                    corrections_count += 1

            # going towards left of the region

            # we are the leftmost region
            if region - 1 == -1:
                region_start = region_indices[region][0]

                # while we are not at the first base of the read
                while region_start != 0:

                    region_start -= 1
                    region_indices[region][0] = region_start
                    corrections_count += 1
            # there is another region in the left side of this region
            if region - 1 != -1:
                region_start, prev_region_end = (
                    region_indices[region][0],
                    region_indices[region - 1][1],
                )
                while region_start - 1 != (prev_region_end):
                    region_start -= 1
                    region_indices[region][0] = region_start
                    corrections_count += 1
        self.assertEqual(corrections_count, 6)

    def test_kmer_correction_counter_ends(self):
        kmer_len = 3
        reads = np.arange(11)
        kmer_counter_list = np.zeros((len(reads) - (kmer_len - 1)), dtype="uint8")
        self.mark_kmer_counter(0, kmer_counter_list, kmer_len, 8, len(reads))

        assert_array_equal(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), kmer_counter_list)

    def mark_kmer_counter(
        self, base_idx, kmer_counter_list, kmer_len, max_kmer_idx, read_length
    ):
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
        while len != 0:
            key += ascii_kmer[len - 1] * multiplier
            multiplier *= 10
            len -= 1

        return key


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
