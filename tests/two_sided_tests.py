import unittest
import numpy as np
from test_modules import copy_kmer, generate_kmers, identify_solid_bases, in_spectrum, transform_to_key, count_occurence, check_solids_cardinality
from test_correction_modules import select_mutations
from numpy.testing import assert_array_equal


class TwoSidedTest(unittest.TestCase):
    def test_correcting_edge_bases(self):
        MAX_LEN = 300
        DEFAULT_KMER_LEN = 13
        spectrum = []
        kmer_length = 4
        # local_read = random.randint(1, 4, 100, dtype="uint8")
        local_read = [2, 1, 3, 3, 3, 2, 2, 3, 4, 2, 1, 4, 3, 1, 1, 2, 3, 1]
        original_read = local_read.copy()
        bases = np.zeros(4, dtype="uint8")
        start, end = 0, len(local_read)
        solids = np.zeros(MAX_LEN, dtype="int8")
        rpossible_base_mutations = np.zeros(10, dtype='uint8')
        lpossible_base_mutations = np.zeros(10, dtype='uint8')
        aux_kmer1 = np.zeros(DEFAULT_KMER_LEN, dtype='uint8')
        aux_kmer2 = np.zeros(DEFAULT_KMER_LEN, dtype='uint8')
        seqlen = end - start
        size = seqlen - kmer_length
        max_iters = 2
        # seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        # local_read[0] = 1
        # generate_kmers(local_read, kmer_length, spectrum)
        spectrum = count_occurence(spectrum)
        #create untrusted kmers within the untrusted base
        local_read[0] = 4
        local_read[16] = 4
        # local_read[10] = 2
        # local_read[11] = 3
        # local_read[12] = 2
        print(spectrum)
        
        for idx in range(max_iters):
            num_corrections = self.correct_core_two_sided(aux_kmer1, aux_kmer2, seqlen, spectrum, kmer_length, bases, solids, local_read, lpossible_base_mutations, rpossible_base_mutations, size)
            if num_corrections == 0:
                print(f"The read is already deemed to be not erroneous at iter index: {idx}")
            elif num_corrections == -1:
                print(f"The read has more than one error within a kmer at iter index: {idx}")
            elif num_corrections == 1:
                print(f"A successful correction is done within the read at iter index: {idx}")
        assert_array_equal(local_read, original_read)
    def correct_core_two_sided(self, ascii_kmer, aux_kmer, seqlen, kmer_spectrum, kmer_len, bases, solids, local_read, lpossible_base_mutations, rpossible_base_mutations, size):
        for i in range(seqlen):
            solids[i] = -1

        # identify whether base is solid or not
        identify_solid_bases(
            local_read, kmer_len, kmer_spectrum, solids, seqlen - kmer_len, aux_kmer
        )
        #check whether solids array does not contain -1, return 0 if yes
        if check_solids_cardinality(solids, seqlen):
            return 0

        klen_idx = kmer_len - 1
        for ipos in range(0, size + 1):
            lpos = 0

            #trusted base
            if solids[ipos] == 1:
                continue

            if ipos >= 0:
                copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
            if ipos >= kmer_len:
                copy_kmer(aux_kmer, local_read, ipos - klen_idx, ipos + 1)
                lpos = -1
            else: 
                copy_kmer(aux_kmer, local_read, 0, kmer_len)
                lpos = ipos
            
            #select all possible mutations for rkmer
            (rnum_bases, _) = select_mutations(kmer_spectrum, bases, ascii_kmer, kmer_len, 0)
                
            #select all possible mutations for lkmer
            (lnum_bases, _)= select_mutations(kmer_spectrum, bases, aux_kmer, kmer_len, lpos)

            i = 0
            num_corrections = 0
            potential_base = -1
            while(i < rnum_bases and num_corrections <= 1):
                rbase = rpossible_base_mutations[i]
                j = 0 
                while (j < lnum_bases):
                    lbase = lpossible_base_mutations[j]
                    #add the potential correction
                    if lbase == rbase:
                        num_corrections += 1
                        potential_base = rbase
                    j += 1
                i += 1
            #apply correction to the current base and return from this function
            if num_corrections == 1 and potential_base != -1:
                local_read[ipos] = potential_base
                return 1
            else:
                print(f"Correction is not successful for index: {ipos} corrections found != 1. Corrections found: {num_corrections}")
        #endfor  0 < seqlen - klen
    
        #for bases > (end - start) - klen)
        for ipos in range(size + 1, seqlen):
            if solids[ipos] == 1:
                continue
            rkmer = local_read[size:]
            lkmer = local_read[ipos - klen_idx: ipos + 1]
            #select all possible base mutations for right kmer
            rnum_bases  = 0
            for base in bases:
                rkmer[ipos - size] = base
                candidate_kmer = transform_to_key(rkmer, kmer_len)
                print(f"Right kmer: {candidate_kmer} in ipos: {ipos}")
                if in_spectrum(kmer_spectrum, candidate_kmer):
                    rpossible_base_mutations[rnum_bases] = base
                    print(f"Yeey its in spectrum,  Right Kmer: {candidate_kmer} in pos {ipos}")
                    rnum_bases += 1
            #select all possible base mutations for left kmer
            lnum_bases = 0
            for base in bases:
                lkmer[-1] = base
                candidate_kmer = transform_to_key(lkmer, kmer_len)
                print(f"Left kmer: {candidate_kmer} in ipos: {ipos}")
                if in_spectrum(kmer_spectrum, candidate_kmer):
                    lpossible_base_mutations[lnum_bases] = base
                    print(f"Yeey its in spectrum,  Left Kmer: {candidate_kmer} in pos {ipos}")
                    lnum_bases += 1
            i = 0
            num_corrections = 0
            potential_base = -1
            while(i < rnum_bases and num_corrections <= 1):
                rbase = rpossible_base_mutations[i]
                j = 0 
                while (j < lnum_bases):
                    lbase = lpossible_base_mutations[j]
                    #add the potential correction
                    if lbase == rbase:
                        num_corrections += 1
                        potential_base = rbase
                    j += 1
                i += 1
            #apply correction to the current base and return from this function
            if num_corrections == 1 and potential_base != -1:
                local_read[ipos] = potential_base
                return 1
            else:
                print(f"Correction is not successful for index: {ipos} corrections found != 1. Corrections found: {num_corrections}")
        return -1

if __name__ == "__main__":
    unittest.main()
