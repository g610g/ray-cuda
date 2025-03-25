import numpy as np
import unittest


class OneSidedTests(unittest.TestCase):
    def test_reverse_complement(self):
        local_read = [1, 2, 3, 4, 1, 2, 2, 1, 4, 1, 2, 2, 5, 5]
        encoded_bases = np.zeros(len(local_read), dtype="uint8")
        copy_kmer(encoded_bases, local_read, 0, len(local_read))
        encode_bases(encoded_bases, len(local_read))
        reverse_comp(encoded_bases, len(local_read))
        if lower(encoded_bases, local_read, len(local_read)):
            print(local_read)
            return
        print(encoded_bases)


def encode_bases(bases, seqlen):
    for idx in range(0, seqlen):
        if bases[idx] == 5:
            bases[idx] = 1


def copy_kmer(aux_kmer, local_read, start, end):
    for i in range(start, end):
        aux_kmer[i - start] = local_read[i]


def reverse_comp(reverse, kmer_len):
    left = 0
    right = kmer_len - 1
    while left <= right:
        comp_left = complement(reverse[left])
        comp_right = complement(reverse[right])
        reverse[left] = comp_right
        reverse[right] = comp_left
        right -= 1
        left += 1


def complement(base):
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


def lower(kmer, aux_kmer, kmer_len):
    for idx in range(0, kmer_len):
        if aux_kmer[idx] > kmer[idx]:
            return False
        elif aux_kmer[idx] < kmer[idx]:
            return True
    return False


if __name__ == "__main__":
    unittest.main()
