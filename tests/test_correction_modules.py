from numba.cuda import target
import numpy as np
from test_modules import (
    identify_trusted_regions,
    in_spectrum,
    transform_to_key,
    give_kmer_multiplicity,
    successor,
    predeccessor

)


def correct_reads_two_sided(
    idx,
    local_reads,
    kmer_len,
    kmer_spectrum,
    bases,
    left_kmer,
    right_kmer,
):
    current_base = local_reads[idx]
    posibility = 0
    candidate = -1

    for alternative_base in bases:
        if alternative_base != current_base:

            # array representation
            left_kmer[-1] = alternative_base
            right_kmer[0] = alternative_base

            # whole number representation
            candidate_left = transform_to_key(left_kmer, kmer_len)
            candidate_right = transform_to_key(right_kmer, kmer_len)

            # the alternative base makes our kmers trusted
            if in_spectrum(kmer_spectrum, candidate_left) and in_spectrum(
                kmer_spectrum, candidate_right
            ):
                posibility += 1
                candidate = alternative_base

    if posibility == 1:
        local_reads[idx] = candidate
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    if posibility == 0:
        pass
        # corrected_counter[threadIdx][counter] = 10
        # counter += 1

    # ignore the correction if more than one possibility
    if posibility > 1:
        pass
        # corrected_counter[threadIdx][counter] = posibility
        # counter += 1

    # return counter


def correct_read_one_sided_right(
    local_read,
    region_end,
    kmer_spectrum,
    kmer_len,
    bases,
    alternatives,
):

    possibility = 0
    alternative = -1

    target_pos = region_end + 1
    ipos = target_pos - (kmer_len - 1)

    forward_kmer = local_read[ipos: target_pos + 1]
    print(forward_kmer)

    # foreach alternative base
    for alternative_base in bases:
        forward_kmer[-1] = alternative_base
        candidate_kmer = transform_to_key(forward_kmer, kmer_len)

        # if the candidate kmer is in the spectrum and has addition evidence that alternative base in trusted as correction by assessing neighbor kmers
        if in_spectrum(kmer_spectrum, candidate_kmer):

            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )
            possibility += 1
            alternative = alternative_base

    print(f"Possibility: {possibility}")
    # returning false will should cause the caller to break the loop since it fails to correct (base on the Musket paper)
    if possibility == 0:
        return False

    # not sure if correct indexing for reads
    if possibility == 1:

        local_read[region_end + 1] = alternative
        print(f"is potential correction (orientation right) -> {target_pos}: {True}, Base: {alternative}")
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            is_potential_correction = successor(
                kmer_len,
                local_read,
                kmer_spectrum,
                alternatives[idx][0],
                2,
                ipos
            )
            print(f"is potential correction (orientation right) -> {target_pos}: {is_potential_correction}, Base: {alternatives[idx][0]}")
            if is_potential_correction:
                if alternatives[idx][1] > choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            return True
        return False


def correct_read_one_sided_left(
    local_read,
    region_start,
    kmer_spectrum,
    kmer_len,
    bases,
    alternatives,
):

    possibility = 0
    alternative = -1

    target_pos = region_start - 1
    backward_kmer = local_read[target_pos : target_pos + kmer_len]
    # If end kmer of trusted region is at the spectrum and when sliding the window, the result kmer is not trusted, then we assume that the end base of that kmer is the sequencing error
    for alternative_base in bases:
        backward_kmer[0] = alternative_base
        candidate_kmer = transform_to_key(backward_kmer, kmer_len)

        # if the candidate kmer is in the spectrum and has addition evidence that alternative base in trusted as correction by assessing neighbor kmers
        if in_spectrum(kmer_spectrum, candidate_kmer):
            # alternative base and its corresponding kmer count
            alternatives[possibility][0], alternatives[possibility][1] = (
                alternative_base,
                give_kmer_multiplicity(kmer_spectrum, candidate_kmer),
            )
            possibility += 1
            alternative = alternative_base
    print(f"Possibility: {possibility}")
    # returning false should cause the caller to break the loop since it fails to correct (base on the Musket paper)
    if possibility == 0:
        return False

    # not sure if correct indexing for reads
    if possibility == 1:

        local_read[target_pos] = alternative
        return True

    # we have to iterate the number of alternatives and find the max element
    if possibility > 1:
        choosen_alternative_base = -1
        choosen_alternative_base_occurence = -1

        for idx in range(possibility):
            is_potential_correction = predeccessor(
                kmer_len,
                local_read,
                kmer_spectrum,
                target_pos,
                alternatives[idx][0],
                2,
            )
            print(f"is potential correction (orientation left) -> {target_pos}: {is_potential_correction} alternative base: {alternatives[idx][0]}")
            if is_potential_correction:
                if alternatives[idx][1] > choosen_alternative_base_occurence:
                    choosen_alternative_base = alternatives[idx][0]
                    choosen_alternative_base_occurence = alternatives[idx][1]

        if choosen_alternative_base_occurence != -1 and choosen_alternative_base != -1:
            local_read[target_pos] = choosen_alternative_base
            return True
        return False
def identify_trusted_regions_v2(local_read, kmer_len, spectrum, seq_len, size):
    MAX_LEN = 300
    left_kmer, right_kmer = -1, -1 
    solid_region = False
    solids = np.zeros(MAX_LEN, dtype='int8')
    regions = np.zeros((10, 2), dtype='int16')
    regions_count = 0
    for pos in range(seq_len):
        solids[pos] = -1

    for ipos in range(size + 1):
        ascii_kmer = local_read[ipos: ipos + kmer_len]
        kmer = transform_to_key(ascii_kmer, kmer_len)
        if in_spectrum(spectrum, kmer):
            if not solid_region:
                solid_region = True
                left_kmer = right_kmer = ipos
            else:
                right_kmer += 1
            for idx in range(ipos, ipos + kmer_len):
                solids[idx] = 1
        else:
            if left_kmer >= 0 :
                regions[regions_count][0], regions[regions_count][1] = left_kmer, right_kmer
                regions_count += 1
                left_kmer = right_kmer = -1
            solid_region = False
    if solid_region and left_kmer >= 0:
        regions[regions_count][0], regions[regions_count][1] = left_kmer, right_kmer
        regions_count += 1
    print(f"Left kmer: {left_kmer} Right kmer : {right_kmer}") 
    return (regions_count, regions.copy(), solids.copy())

def one_sided_v2(local_read, kmer_len, size, seq_len, spectrum, solids, bases, max_corrections):
    region_indices = np.zeros((10, 2), dtype='uint16')
    regions_count = identify_trusted_regions(0, seq_len, spectrum, local_read, kmer_len, region_indices, solids)

    for region in range(regions_count):
        best_base = -1
        best_base_occurence = -1
        right_mer_idx = region_indices[region][1]
        last_position = -1
        num_corrections = 0
        original_read = local_read[right_mer_idx + 1:]
        for target_pos in range(right_mer_idx + 1, size + 1):
            done = False
            spos = target_pos - (kmer_len - 1)
            ascii_kmer = local_read[spos: target_pos + 1]
            if solids[target_pos] == 1:
                print(f"Base at base index {target_pos} is trusted")
                break
            (num_bases, alternative_bases) = select_mutatations(spectrum, bases, ascii_kmer, kmer_len, -1)

            #directly apply correction if mutation is equal to 1
            if num_bases == 1:
                local_read[target_pos] = alternative_bases[0]
                done = True
            else:
                for idx in range(0, num_bases):
                    if successor(kmer_len, local_read, spectrum, alternative_bases[idx], 2, spos):
                        aux_ascii_kmer = local_read[spos: target_pos + 1]
                        aux_ascii_kmer[-1] = alternative_bases[idx]
                        aux_kmer = transform_to_key(aux_ascii_kmer, kmer_len)
                        aux_occurence = give_kmer_multiplicity(spectrum, aux_kmer)
                        if aux_occurence > best_base_occurence:
                            best_base_occurence = aux_occurence
                            best_base = alternative_bases[idx]
                #apply correction
                if best_base_occurence != -1 and best_base != -1:
                    local_read[target_pos] = best_base
                    done = True

            #check how many corrections is done
            if done:
                region_indices[region][1] = target_pos
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    #revert back reads
                    if num_corrections > max_corrections:
                        print(f"Reverting corrections made from index: {last_position} to index: {target_pos}")

                        for pos in range(last_position, target_pos + 1):
                            print(f"Reverting base position: {pos}")
                            local_read[pos] = original_read[pos - (right_mer_idx + 1)]
                        region_indices[region][1] =  last_position - 1
                else:
                    last_position = target_pos
                    num_corrections = 0
                continue
            break
    
        #endfor rkmer_idx + 1 to size 
        
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:

            last_position = -1
            num_corrections = 0
            original_read = local_read[0: lkmer_idx]

    #endfor regions_count

def select_mutatations(spectrum, bases, ascii_kmer, kmer_len, pos):
    num_bases = 0 
    bases = []
    for base in bases:
        ascii_kmer[pos] = base
        candidate = transform_to_key(ascii_kmer,kmer_len)
        if in_spectrum(spectrum, candidate):
            num_bases += 1
            bases.append(base)
    return (num_bases, bases)
