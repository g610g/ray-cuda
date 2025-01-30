import unittest
import numpy as np
from numpy import random
from test_modules import generate_kmers, identify_solid_bases, in_spectrum, transform_to_key, count_occurence
from test_correction_modules import correct_reads_two_sided
from numpy.testing import assert_array_equal


class TwoSidedTest(unittest.TestCase):
    def test_correcting_edge_bases(self):
        MAX_LEN = 300
        spectrum = []
        kmer_length = 4
        # local_read = random.randint(1, 4, 100, dtype="uint8")
        local_read = [2, 1, 2, 3, 4, 1, 2, 3, 4, 2, 1, 2, 3, 4, 1, 2, 3, 1]
        original_read = local_read.copy()
        bases = np.zeros(4, dtype="uint8")
        start, end = 0, len(local_read)
        solids = np.zeros(MAX_LEN, dtype="int8")
        rpossible_base_mutations = np.zeros(10, dtype='uint8')
        lpossible_base_mutations = np.zeros(10, dtype='uint8')

        size = (end - start)  - kmer_length
        # seeding solids with -1s
        for idx in range(len(local_read)):
            solids[idx] = -1

        # seeding bases
        for idx in range(4):
            bases[idx] = idx + 1

        generate_kmers(local_read, kmer_length, spectrum)
        spectrum = count_occurence(spectrum)
        print(spectrum)
        local_read[2] = 3

        # modify random bases in the local read
        # for _idx in range(4):
        #     random_idx = random.randint(0, 70)
        #     local_read[random_idx] = random.randint(1, 4)

        identify_solid_bases(
            local_read, 0, len(local_read), kmer_length, spectrum, solids
        )

        print(solids)
        klen_idx = kmer_length - 1
        for ipos in range(0, size + 1):
            lpos = 0
            if ipos >= 0:
                rkmer = local_read[ipos: ipos + kmer_length] 
            if ipos >= kmer_length:
                lkmer = local_read[ipos - klen_idx: ipos + 1]
                lpos = -1
            else: 
                lkmer = local_read[0: kmer_length]
                lpos = ipos
            #trusted base
            if solids[ipos] == 1:
                continue

                #do corrections right here
                
                #select all possible mutations for rkmer
            rnum_bases = 0
            roriginal_base = rkmer[0]
            for base in bases:
                rkmer[0] = base
                candidate_kmer = transform_to_key(rkmer, kmer_length)
                if in_spectrum(spectrum, candidate_kmer):
                    rpossible_base_mutations[rnum_bases] = base
                    rnum_bases += 1 
                    
                #select all possible mutations for lkmer
            lnum_bases = 0
            loriginal_base = lkmer[lpos]
            for base in bases:
                lkmer[lpos] = base
                candidate_kmer = transform_to_key(lkmer, kmer_length)
                if in_spectrum(spectrum, candidate_kmer):
                    lpossible_base_mutations[lnum_bases] = base
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
                #apply correction to the current base
            if num_corrections == 1 and potential_base != -1:
                local_read[ipos] = potential_base
            
            #endfor  0 < seqlen - klen
            
            #for bases > (end - start) - klen)

        rkmer = local_read[size:]
        for ipos in range(size + 1, end - start):
            lkmer = local_read[ipos - klen_idx: ipos + 1]
            if solids[ipos] == 1:
                continue
            #select mutations for right kmer
            rnum_bases  = 0
            for base in bases:
                rkmer[ipos - size] = base
                candidate_kmer = transform_to_key(rkmer, kmer_length)
                if in_spectrum(spectrum, candidate_kmer):
                    rpossible_base_mutations[rnum_bases] = base
                    rnum_bases += 1
            for base in bases:
                lkmer[-1] = base
                candidate_kmer = transform_to_key(rkmer, kmer_length)
                if in_spectrum(spectrum, candidate_kmer):
                    lpossible_base_mutations[lnum_bases] = base
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
                #apply correction to the current base
            if num_corrections == 1 and potential_base != -1:
                local_read[ipos] = potential_base
    
        assert_array_equal(local_read, original_read)


if __name__ == "__main__":
    unittest.main()
