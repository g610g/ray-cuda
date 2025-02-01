import unittest
import numpy as np
from numpy import random, size
from numpy.testing import assert_array_equal
from test_modules import (
    count_occurence,
    identify_trusted_regions,
    generate_kmers,
    in_spectrum
)
from test_correction_modules import (
    correct_read_one_sided_right,
    correct_read_one_sided_left,
    identify_trusted_regions_v2,
)


class OneSidedTests(unittest.TestCase):
    def test_one_sided_core(self):
        MAX_LEN = 300
        spectrum = []
        kmer_length = 4
        local_read = [
            4,
            2,
            1,
            4,
            1,
            1,
            2,
            5,
            3,
            2,
            3,
            1,
            2,
            4,
            3,
            1,
            2,
            2,
            4,
            3,
            2,
            1,
            2,
            3,
            4,
            1,
            4,
            2,
        ]
        # local_read = random.randint(1, 4, 100, dtype="uint8")
        original_read = local_read.copy()

        # generates dummy kmers
        # for idx in range(10):
        #     dummy_read = random.randint(1, 4, 100, dtype="uint8")
        #     generate_kmers(dummy_read, kmer_length, spectrum)
        generate_kmers(local_read, kmer_length, spectrum)
        spectrum = count_occurence(spectrum)
        # array used to store base alternative and its corresponding kmer count
        alternatives = np.zeros((4, 2), dtype="uint32")
        local_read_len = len(local_read)
        num_kmers = len(local_read) - (kmer_length - 1)
        # array that keep tracks of corrections made for every kmer
        bases = np.zeros(4, dtype="uint8")
        start, end = 0, len(local_read)
        region_indices = np.zeros((10, 2), dtype="int8")
        solids = np.zeros(MAX_LEN, dtype="int8")
        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1
        print(local_read)
        spectrum.append([1242, 1])
        spectrum.append([3213, 1])
        spectrum.append([3413, 1])
        local_read[14] = 4
        local_read[22] = 4
        local_read[26] = 2
        local_read[1] = 3

        # run one sided for a number of time
        max_correction = 4
        for _ in range(max_correction):
            # seeding solids with -1s
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
            print(solids)
            print(region_indices)
            if regions_count == 0:
                return

            # no unit tests for this part yet
            for region in range(regions_count):
                # going towards right of the region

                # there is no next region
                if region == (regions_count - 1):
                        print("Entering for region going right that has no next region")
                        region_end = region_indices[region][1]
                        original_region_end = region_end
                        corrections_count = 0
                        last_position = -1
                        original_bases = local_read[region_end + 1 :]
                        # while we are not at the end base of the read
                        while region_end != (end - start) - 1:
                            target_pos = region_end + 1
                            # keep track the number of corrections made here
                            if not correct_read_one_sided_right(
                                local_read,
                                region_end,
                                spectrum,
                                kmer_length,
                                bases,
                                alternatives,
                            ):
                                print(f"Correcting towards right for index {target_pos} is not successful") 
                                break

                            # extend the portion of region end for successful correction and check the number of corrections done
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if target_pos - last_position < kmer_length:
                                    corrections_count += 1

                                    # revert back bases from last position to target position and revert region
                                    # prevents cumulative incorrect corrections
                                    if corrections_count > max_correction:
                                        print(f"Index from {last_position} to {target_pos} have exceed maximum corrections")
                                        for pos in range(last_position, target_pos + 1):
                                            local_read[pos] = original_bases[
                                                pos - last_position
                                            ]
                                        region_indices[region][1] = original_region_end
                                        print("stopping correction")
                                        break
                                    region_end += 1
                                    region_indices[region][1] = target_pos
                                # recalculate the last position 
                                else:
                                    print("Resetting last position in correction orientation right")
                                    last_position = target_pos
                                    original_region_end = target_pos
                                    corrections_count = 0
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                # there is a next region
                elif region != (regions_count - 1):
                        print("Entering for region going right that has next region")
                        region_end = region_indices[region][1]
                        original_region_end = region_end
                        next_region_start = region_indices[region + 1][0]
                        original_bases = local_read[region_end + 1 :]
                        last_position = -1
                        corrections_count = 0
                        # the loop will not stop until it does not find another region
                        while region_end != (next_region_start - 1):
                            target_pos = region_end + 1
                            if not correct_read_one_sided_right(
                                local_read,
                                region_end,
                                spectrum,
                                kmer_length,
                                bases,
                                alternatives,
                            ):
                                # fails to correct this region and on this orientation
                                print(f"Correting towards right for index {target_pos} is not successful") 
                                break

                            # extend the portion of region end for successful correction
                            else:

                                if last_position < 0:
                                    last_position = target_pos
                                if target_pos - last_position < kmer_length:
                                    corrections_count += 1

                                    # revert back bases from last position to target position and stop correction for this region oritentation
                                    if corrections_count > max_correction:
                                        print(f"Index from {last_position} to {target_pos} have exceed maximum corrections")
                                        for pos in range(last_position, target_pos + 1):
                                            local_read[pos] = original_bases[
                                                pos - last_position
                                            ]
                                        region_indices[region][1] = original_region_end
                                        print("stopping correction")
                                        break
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                                # recalculate the last position and reset corrections made
                                else:
                                    print("Resetting last position in correction orientation right")
                                    last_position = target_pos
                                    original_region_end = target_pos
                                    corrections_count = 0
                                    region_end += 1
                                    region_indices[region][1] = target_pos

                # going towards left of the region
                # we are the leftmost region
                if region == 0:
                        print("Entering for region going left that is the leftmost region")
                        region_start = region_indices[region][0]
                        original_region_start = region_start
                        last_position = -1
                        corrections_count = 0
                        original_bases = local_read[0 : region_start + 1]

                        # while we are not at the first base of the read
                        while region_start > 0:
                            target_pos = region_start - 1
                            if not correct_read_one_sided_left(
                                local_read,
                                region_start,
                                spectrum,
                                kmer_length,
                                bases,
                                alternatives,
                            ):

                                print(f"Correcting towards left for index {target_pos} is not successful") 
                                break
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if last_position - target_pos < kmer_length:
                                    corrections_count += 1
                                    # revert back bases
                                    if corrections_count > max_correction:
                                        print(f"Index from {target_pos} to {last_position} have exceed maximum corrections")
                                        for pos in range(target_pos, last_position + 1):
                                            local_read[pos] = original_bases[pos]
                                        region_indices[region][0] = original_region_start
                                        print("stopping correction")
                                        break
                                    region_start -= 1
                                    region_indices[region][0] = region_start
                                else:
                                    print("Resetting last position in correction orientation left")
                                    last_position = target_pos
                                    original_region_start = target_pos
                                    corrections_count = 0
                                    region_start = target_pos
                                    region_indices[region][0] = region_start

                # there is another region in the left side of this region
                elif region > 0:
                        print("Entering for region going left that has preceeding next region")
                        region_start, prev_region_end = (
                            region_indices[region][0],
                            region_indices[region - 1][1],
                        )
                        original_region_start = region_start
                        last_position = -1
                        corrections_count = 0
                        original_bases = local_read[0 : region_start + 1]

                        while region_start - 1 > (prev_region_end):
                            target_pos = region_start - 1
                            if not correct_read_one_sided_left(
                                local_read,
                                region_start,
                                spectrum,
                                kmer_length,
                                bases,
                                alternatives,
                            ):
                                print(f"Correcting towards left for index {target_pos} is not successful") 
                                break
                            else:
                                if last_position < 0:
                                    last_position = target_pos
                                if last_position - target_pos < kmer_length:
                                    corrections_count += 1
                                    # revert back bases
                                    if corrections_count > max_correction:
                                        print(f"Index from {target_pos} to {last_position} have exceed maximum corrections")
                                        for pos in range(target_pos, last_position + 1):
                                            local_read[pos] = original_bases[pos]
                                        region_indices[region][0] = original_region_start
                                        print("stopping correction")
                                        break
                                    region_start -= 1
                                    region_indices[region][0] = region_start
                                else:
                                    print("Resetting last position in correction orientation left")
                                    last_position = target_pos
                                    original_region_start = target_pos
                                    corrections_count = 0
                                    region_start -= 1
                                    region_indices[region][0] = region_start

        # endfor

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
        assert_array_equal(original_read, local_read)

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

    def test_region_identification(self):
        local_read = [1, 2, 3, 3, 3, 2, 1, 2, 4, 1, 2, 3, 1, 2, 3, 4, 5]
        kmer_len = 4
        seq_len = len(local_read)
        size = seq_len - kmer_len
        spectrum = []
        generate_kmers(local_read, kmer_len, spectrum)
        spectrum = count_occurence(spectrum)

        local_read[0] = 3
        local_read[1] = 4
        local_read[7] = 1

        (regions_count, regions, solids)  = identify_trusted_regions_v2(local_read, kmer_len, spectrum, seq_len, size)
        print(f"regions count:{regions_count} regions:{regions}")
        print(f"Solids: {solids}")

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
