import numpy as np
from test_modules import (
    forward_base,
    identify_trusted_regions,
    identify_trusted_regions_host,
    in_spectrum,
    predeccessor_v2,
    predeccessor_v2_host,
    successor_host,
    transform_to_key,
    give_kmer_multiplicity,
    successor,
    copy_kmer,
    backward_base,
    transform_to_key_host
)

def identify_trusted_regions_v2(local_read,ascii_kmer,  kmer_len, spectrum, seq_len, size):
    MAX_LEN = 300
    left_kmer, right_kmer = -1, -1 
    solid_region = False
    solids = np.zeros(MAX_LEN, dtype='int8')
    regions = np.zeros((10, 2), dtype='int16')
    regions_count = 0
    for pos in range(seq_len):
        solids[pos] = -1

    for ipos in range(size + 1):
        copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
        kmer = transform_to_key(ascii_kmer, kmer_len)
        if in_spectrum(spectrum, kmer):
            if not solid_region:
                solid_region = True
                left_kmer = right_kmer = ipos
            else:
                right_kmer += 1
            print(f"Marking in index {ipos} to {ipos + (kmer_len - 1)}")
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

def one_sided_v2(local_read, aux_corrections , ascii_kmer, aux_kmer, kmer_len,seq_len, spectrum, solids, bases, max_corrections, distance):
    #corrections_count keep tracks the number of position corrections made
    corrections_count = 0
    region_indices = np.zeros((10, 2), dtype='uint16')
    regions_count = identify_trusted_regions(seq_len, spectrum, local_read, kmer_len, region_indices, solids, aux_kmer)
    print(solids)
    print(region_indices[:regions_count,:])
    print(f"Regions Count: {regions_count}")
    # print(region_indices) 
    for region in range(regions_count):
        right_mer_idx = region_indices[region][1]
        right_orientation_idx = -1
        last_position = -1
        num_corrections = 0
        for target_pos in range(right_mer_idx + 1, seq_len):
            best_base = -1
            best_base_occurence = -1
            done = False
            spos = target_pos - (kmer_len - 1)

            #get the ascii kmer
            if target_pos == right_mer_idx + 1:
                copy_kmer(ascii_kmer, local_read, spos, target_pos + 1)
            else:
                forward_base(ascii_kmer, local_read[target_pos], kmer_len)

            if solids[target_pos] == 1:
                print(f"Base at base index {target_pos} is trusted")
                break
            (num_bases, alternative_bases) = select_mutations(spectrum, bases, ascii_kmer, kmer_len, kmer_len - 1)

            #directly apply correction if mutation is equal to 1
            if num_bases == 1:
                print(f"correction toward right index {target_pos} has one alternative base: {alternative_bases[0]}")

                if right_orientation_idx < 0:
                    print(f"index {target_pos} is the first index in right orientation that has been corrected")
                    right_orientation_idx = target_pos
                aux_corrections[target_pos] = alternative_bases[0]
                ascii_kmer[kmer_len - 1] = alternative_bases[0]
                corrections_count += 1
                done = True
            else:
                print(f"alternative bases: {alternative_bases}")
                print(f"Num of alternative bases: {num_bases}")
                for idx in range(0, num_bases):
                    copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                    if successor(kmer_len, local_read, aux_kmer, spectrum, alternative_bases[idx], spos, distance):
                        print("successor returns true")
                        copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                        aux_kmer[kmer_len - 1] = alternative_bases[idx]
                        candidate = transform_to_key(aux_kmer, kmer_len)
                        aux_occurence = give_kmer_multiplicity(spectrum, candidate)

                        if aux_occurence > best_base_occurence:
                            best_base_occurence = aux_occurence
                            best_base = alternative_bases[idx]
                    else:
                        print("successor returns false")

                #apply correction
                if best_base_occurence > -1 and best_base > -1:
                    print(f"{best_base} is the chosen alternative base")
                    if right_orientation_idx < 0:
                        print(f"index {target_pos} is the first index in right orientation that has been corrected")
                        right_orientation_idx = target_pos

                    aux_corrections[target_pos] = best_base
                    ascii_kmer[kmer_len - 1] = best_base
                    corrections_count += 1
                    done = True

            #check how many corrections is done and extend the region index
            if done:
                region_indices[region][1] = target_pos
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    #remove recorded corrections if num corrections exceeds max corrections
                    if num_corrections > max_corrections:
                        print(f"Reverting corrections made from index: {last_position} to index: {target_pos}, max corrections is: {max_corrections} num corrections is :{num_corrections}")
                        for pos in range(last_position, target_pos + 1):
                            print(f"Reverting base position: {pos}, base: {aux_corrections[pos]}")
                            aux_corrections[pos] = 0

                        #remove corrections in aux_corrections
                        corrections_count -= num_corrections
                        region_indices[region][1] =  last_position - 1
                        break
                        #break correction for this orientation after reverting back kmers
                else:
                    #modify original read elements right here
                    last_position = target_pos
                    num_corrections = 0

                continue
            print(f"Orientation to the right index: {target_pos} breaks (done == False) ")
            print(f"best base: {best_base} best base occurence: {best_base_occurence}")

            #break correction if done is False
            break

        #endfor rkmer_idx + 1 to seq_len
        #for left orientation
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:
            if right_orientation_idx >= 0 :
                last_position = right_orientation_idx
            else:
                last_position = -1
            print(f" lkmer: {lkmer_idx} last position for left orientation is {last_position}")
            num_corrections = 0

            for pos in range(lkmer_idx - 1, -1, -1):
                #the current base is trusted
                if solids[pos] == 1:
                    break
                best_base = -1
                best_base_occurence = -1
                done = False
                if pos == lkmer_idx - 1:
                    copy_kmer(ascii_kmer, local_read, pos, pos + kmer_len)
                else:
                    backward_base(ascii_kmer, local_read[pos], kmer_len)

                (num_bases, alternative_bases) = select_mutations(spectrum, bases, ascii_kmer, kmer_len, 0)

                print(f"alternative bases for left orientation index from {pos} to {pos + (kmer_len - 1)} is {alternative_bases}")
                #apply correction 
                if num_bases == 1:
                    print(f"correction toward left index {pos} has one alternative base: {alternative_bases[0]}")
                    aux_corrections[pos] = alternative_bases[0]
                    ascii_kmer[0] = alternative_bases[0]
                    corrections_count += 1
                    done = True
                else:
                    for idx in range(num_bases):
                        copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                        if predeccessor_v2(kmer_len, local_read, aux_kmer, spectrum, pos, alternative_bases[idx], distance):
                            print(f"predeccessor returns true for base {alternative_bases[idx]}")
                            copy_kmer(aux_kmer, ascii_kmer, 0, kmer_len)
                            aux_kmer[0] = alternative_bases[idx]
                            candidate = transform_to_key(aux_kmer, kmer_len)
                            aux_occurence = give_kmer_multiplicity(spectrum, candidate)
                            if aux_occurence > best_base_occurence:
                                best_base_occurence = aux_occurence
                                best_base = alternative_bases[idx]
                        else:
                            print("predeccessor returns False")

                    if best_base > 0 and best_base_occurence > 0:
                        print(f"best base is {best_base}")
                        aux_corrections[pos] = best_base
                        ascii_kmer[0] = best_base
                        corrections_count += 1
                        done = True
                #checking corrections that have done
                if done:
                    region_indices[region][0] = pos
                    if last_position < 0:
                            last_position = pos
                    if last_position - pos < kmer_len:
                        num_corrections += 1
                        #remove recorded corrections if num corrections exceeds max corrections
                        if num_corrections > max_corrections:
                            print(f"reverting corrections made from index:{pos} to index:{last_position} , max corrections is:{max_corrections} num corrections is: {num_corrections}")
                            for base_idx in range(pos, last_position + 1):
                                print(f"Reverting base position: {base_idx} removing base {aux_corrections[base_idx]}")
                                aux_corrections[base_idx] = 0
                            corrections_count -= num_corrections
                            region_indices[region][0] = last_position + 1
                            break
                    else:
                        last_position = pos
                        num_corrections = 0
                    continue

                #the correction for the current base is done == False
                print(f"Orientation to the left index: {pos} breaks (done == False) ")
                break
            #endfor lkmer_idx to 0

    #endfor regions_count
    return corrections_count
def select_mutations(spectrum, bases, ascii_kmer, kmer_len, pos):
    num_bases = 0 
    selected_bases = []
    original_base = ascii_kmer[pos]
    for base in bases:
        if  original_base == base:
            #print(f"original base {origi}")
            continue
        ascii_kmer[pos] = base
        candidate = transform_to_key_host(ascii_kmer)
        if in_spectrum(spectrum, candidate):
            num_bases += 1
            selected_bases.append(base)

    ascii_kmer[pos] = original_base
    return (num_bases, selected_bases.copy())

def one_sided_v2_host(local_read, original_read, ascii_kmer, aux_kmer, kmer_len,seq_len, spectrum, solids, bases, max_corrections, distance):
    #corrections_count keep tracks the number of position corrections made
    region_indices = np.zeros((10, 2), dtype='uint16')
    regions_count = identify_trusted_regions_host(seq_len, spectrum, local_read, kmer_len, region_indices, solids, aux_kmer)
    print(solids)
    # print(region_indices) 
    for region in range(regions_count):
        right_mer_idx = region_indices[region][1]
        last_position = -1
        num_corrections = 0
        for target_pos in range(right_mer_idx + 1, seq_len):
            best_base = -1
            best_base_occurence = -1
            done = False
            spos = target_pos - (kmer_len - 1)

            #get the ascii kmer
            if target_pos == right_mer_idx + 1:
                ascii_kmer = local_read[spos: target_pos + 1].copy()
            else:
                ascii_kmer = original_read[spos: target_pos + 1].copy()
            if solids[target_pos] == 1:
                print(f"Base at base index {target_pos} is trusted")
                break
            (num_bases, alternative_bases) = select_mutations(spectrum, bases, ascii_kmer, kmer_len, kmer_len - 1)

            #directly apply correction if mutation is equal to 1
            if num_bases == 1:
                print(f"correction toward right index {target_pos} has one alternative base: {alternative_bases[0]}")
                original_read[target_pos] = alternative_bases[0]
                done = True
            else:
                print(f"alternative bases: {alternative_bases}")
                print(f"Num of alternative bases: {num_bases}")
                for idx in range(0, num_bases):
                    if successor_host(kmer_len, local_read, aux_kmer, spectrum, alternative_bases[idx], spos, distance, seq_len):
                        print("successor returns true")
                        aux_kmer = ascii_kmer.copy()
                        aux_kmer[kmer_len - 1] = alternative_bases[idx]
                        candidate = transform_to_key_host(aux_kmer)
                        aux_occurence = give_kmer_multiplicity(spectrum, candidate)

                        if aux_occurence > best_base_occurence:
                            best_base_occurence = aux_occurence
                            best_base = alternative_bases[idx]
                    else:
                        print("successor returns false")

                #apply correction
                if best_base_occurence > -1 and best_base > -1:
                    print(f"{best_base} is the chosen alternative base")
                    original_read[target_pos] = best_base
                    done = True

            #check how many corrections is done and extend the region index
            if done:
                region_indices[region][1] = target_pos
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    #remove recorded corrections if num corrections exceeds max corrections
                    if num_corrections > max_corrections:
                        print(f"Reverting corrections made from index: {last_position} to index: {target_pos}, max corrections is: {max_corrections} num corrections is :{num_corrections}")
                        for pos in range(last_position, target_pos + 1):
                            print(f"Reverting base position: {pos}, base: {original_read[pos]}")
                            original_read[pos] = local_read[pos]

                        region_indices[region][1] =  last_position - 1
                        break
                        #break correction for this orientation after reverting back kmers
                else:
                    #modify original read elements right here
                    last_position = target_pos
                    num_corrections = 0

                continue
            print(f"Orientation to the right index: {target_pos} breaks (done == False) ")
            print(f"best base: {best_base} best base occurence: {best_base_occurence}")

            #break correction if done is False
            break

        #endfor rkmer_idx + 1 to seq_len
        #for left orientation
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:
            last_position = -1
            num_corrections = 0
            #original_read = local_read[0:seq_len].copy()
            for pos in range(lkmer_idx - 1, -1, -1):
                #the current base is trusted
                if solids[pos] == 1:
                    break
                best_base = -1
                best_base_occurence = -1
                done = False
                if pos == lkmer_idx - 1:
                    ascii_kmer = local_read[pos: pos + kmer_len].copy()
                else:
                    ascii_kmer = original_read[pos: pos + kmer_len].copy()

                (num_bases, alternative_bases) = select_mutations(spectrum, bases, ascii_kmer, kmer_len, 0)

                print(f"alternative bases for left orientation index from {pos} to {pos + (kmer_len - 1)} is {alternative_bases}")
                #apply correction 
                if num_bases == 1:
                    print(f"correction toward left index {pos} has one alternative base: {alternative_bases[0]}")
                    original_read[pos] = alternative_bases[0]
                    done = True
                else:
                    for idx in range(num_bases):
                        if predeccessor_v2_host(kmer_len, local_read, aux_kmer, spectrum, pos, alternative_bases[idx], distance):
                            print(f"predeccessor returns true for base {alternative_bases[idx]}")

                            aux_kmer = original_read[pos: pos + kmer_len].copy()

                            aux_kmer[0] = alternative_bases[idx]
                            candidate = transform_to_key(aux_kmer, kmer_len)
                            aux_occurence = give_kmer_multiplicity(spectrum, candidate)
                            if aux_occurence > best_base_occurence:
                                best_base_occurence = aux_occurence
                                best_base = alternative_bases[idx]
                        else:
                            print("predeccessor returns False")

                    if best_base > 0 and best_base_occurence > 0:
                        print(f"best base is {best_base}")
                        original_read[pos] = best_base
                        done = True
                #checking corrections that have done
                if done:
                    region_indices[region][0] = pos
                    if last_position < 0:
                            last_position = pos
                    if last_position - pos < kmer_len:
                        num_corrections += 1
                        #remove recorded corrections if num corrections exceeds max corrections
                        if num_corrections > max_corrections:
                            print(f"reverting corrections made from index:{pos} to index:{last_position} , max corrections is:{max_corrections} num corrections is: {num_corrections}")
                            for base_idx in range(pos, last_position + 1):
                                print(f"Reverting base position: {base_idx} removing base {original_read[base_idx]}")
                                original_read[base_idx] = local_read[base_idx]

                            region_indices[region][0] = last_position + 1
                            break
                    else:
                        last_position = pos
                        num_corrections = 0
                    continue

                #the correction for the current base is done == False
                print(f"Orientation to the left index: {pos} breaks (done == False) ")
                break
            #endfor lkmer_idx to 0
    #endfor regions_count
