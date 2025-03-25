import unittest
import numpy as np
from numpy.testing import assert_array_equal

# THIS IS THE TESTING MODULE WHERE I CONDUCT UNIT TESTS ON THE LOGICS THAT WAS IMPLEMENTED ON THE KERNEL LOGIC.
# tHE UNIT TESTING IS SEPERATED AND IS EXECUTED ON A CPU RUNTIME AND NOT ON gpu RUNTIME AS I FOUND IT HARD TO FIND RESOURCES IN ORDER TO UNIT TEST KERNEL CUDA CODES


class CorrectionTests(unittest.TestCase):
    def test_to_array_kmer(self):
        kmer = 1123224414223322412
        km = np.zeros(19, dtype="uint8")
        rep = np.zeros(19, dtype="uint8")
        self.to_array_kmer(km, 19, kmer)
        self.copy_kmer(rep, km, 0, 19)
        self.reverse_comp(rep, 19)
        print(km)
        if self.lower(km, rep):
            print(f"{rep} is lexicographically smaller than {km}")

    def to_array_kmer(self, km, kmer_length, whole_number_km):
        for i in range(kmer_length):
            km[i] = (whole_number_km // (10 ** (kmer_length - 1 - i))) % 10

    def reverse_comp(self, reverse, kmer_len):
        left = 0
        right = kmer_len - 1
        while left <= right:
            comp_left = self.complement(reverse[left])
            comp_right = self.complement(reverse[right])
            reverse[left] = comp_right
            reverse[right] = comp_left
            right -= 1
            left += 1

    def copy_kmer(self, aux_kmer, local_read, start, end):
        for i in range(start, end):
            aux_kmer[i - start] = local_read[i]

    def lower(self, kmer, aux_kmer):
        DEFAULT_KMERLEN = 19
        for idx in range(DEFAULT_KMERLEN):
            if aux_kmer[idx] == kmer[idx]:
                continue
            elif aux_kmer[idx] < kmer[idx]:
                return True
            else:
                return False

        return False

    def complement(self, base):
        if base == 1:
            return 4
        elif base == 2:
            return 3
        elif base == 3:
            return 2
        elif base == 4:
            return 1
        else:
            return 5


if __name__ == "__main__":
    unittest.main()
