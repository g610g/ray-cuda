from host.host_helpers import identify_trusted_regions,copy_kmer, select_mutations, transform_to_key, successor_v2, predeccessor_v2, give_kmer_multiplicity, all_solid_base
import ray
import numpy as np



#entry for one sided
@ray.remote(num_cpus=1)
def entry(reads, kmer_len, spectrum, offsets, num_errors, max_iters):

    #for the assign reads in this core, we will run one sided on a batch of reads
    MAX_LEN = 300
    local_read = np.zeros(MAX_LEN, dtype='uint8')
    original_read = np.zeros(MAX_LEN, dtype='uint8')
    solids = np.zeros(MAX_LEN, dtype='int8')
    region_indices = np.zeros((10, 2), dtype='int32')
    alternative_bases = np.zeros(4, dtype='uint8')
    bases = np.zeros(4, dtype='uint8')
    for idx in range(4):
        bases[idx] = idx + 1

    reads_done = 0
    #for each read
    for offset in offsets:
        start, end = offset[0], offset[1]
        seq_len = end - start

        #copy to local read
        for idx in range(seq_len):
            local_read[idx] = reads[idx + start]
            original_read[idx] = reads[idx + start]

        for nerr in range(1, num_errors + 1):
            distance = num_errors - nerr + 1
            for _ in range(max_iters):

                for idx in range(seq_len):
                    solids[idx] = -1
                original_read = local_read[:seq_len].copy()
                one_sided(local_read, original_read, region_indices, alternative_bases, kmer_len, seq_len, spectrum, solids, bases, nerr, distance)
                local_read = original_read[:seq_len].copy()

        reads_done += 1
        print(f"{reads_done} number of reads is done")
#planning to compile this with numba jit
def one_sided(local_read, original_read, region_indices, selected_bases, kmer_len, seq_len, spectrum, solids, bases, max_corrections, distance ):

    ascii_kmer = np.zeros(kmer_len, dtype='uint8')
    aux_kmer = np.zeros(kmer_len, dtype='uint8')

    size = seq_len - kmer_len
    regions_count = identify_trusted_regions(seq_len, spectrum, local_read, kmer_len, region_indices, solids, aux_kmer, size)

    if all_solid_base(solids, seq_len):
         return 

    for region in range(regions_count):
        right_mer_idx = region_indices[region][1]
        last_position = -1
        num_corrections = 0
        for target_pos in range(right_mer_idx + 1, seq_len):
            if solids[target_pos] == 1:
                break

            best_base = -1
            best_base_occurence = -1
            done = False
            spos = target_pos - (kmer_len - 1)

            if target_pos == right_mer_idx + 1:
                copy_kmer(ascii_kmer, local_read, spos, target_pos + 1)
            else:
                copy_kmer(ascii_kmer, original_read, spos, target_pos + 1)

            #select all possible mutations 
            num_bases = select_mutations(spectrum, bases, ascii_kmer, kmer_len, kmer_len - 1, selected_bases)

            #directly apply correction if mutation is equal to 1
            if num_bases == 1:
                original_read[target_pos] = selected_bases[0]
                done = True
            else:
                for idx in range(0, num_bases):
                    if successor_v2(kmer_len, original_read, aux_kmer, spectrum, selected_bases[idx], spos, distance, seq_len):
                        copy_kmer(aux_kmer, original_read, spos, target_pos + 1)
                        aux_kmer[kmer_len - 1] = selected_bases[idx]
                        candidate = transform_to_key(aux_kmer)
                        aux_occurence = give_kmer_multiplicity(spectrum, candidate)

                        if aux_occurence > best_base_occurence:
                            best_base_occurence = aux_occurence
                            best_base = selected_bases[idx]

                if best_base_occurence > 0 and best_base > 0:
                    original_read[target_pos] = best_base
                    done = True

            #check how many corrections is done and extend the region index
            if done:
                region_indices[region][1] = target_pos
                if last_position < 0:
                    last_position = target_pos
                if target_pos - last_position < kmer_len:
                    num_corrections += 1
                    #revert back reads if corrections exceed max_corrections
                    if num_corrections > max_corrections:
                        for pos in range(last_position, target_pos + 1):
                            original_read[pos] = local_read[pos]
                        region_indices[region][1] =  last_position - 1
                        break
                        #break correction for this orientation after reverting back kmers
                else:
                    #modify original read elements right here
                    last_position = target_pos
                    num_corrections = 0

                continue

            #break correction if done is False
            break

        #endfor rkmer_idx + 1 to seq_len

        #for left orientation
        lkmer_idx = region_indices[region][0]
        if lkmer_idx > 0:
            last_position = -1
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
                    copy_kmer(ascii_kmer, original_read, pos, pos + kmer_len)

                num_bases = select_mutations(spectrum, bases, ascii_kmer, kmer_len, 0, selected_bases)

                # apply correction 
                if num_bases == 1:
                    original_read[pos] = selected_bases[0]
                    done = True
                else:
                    for idx in range(0, num_bases):
                        if predeccessor_v2(kmer_len, original_read, aux_kmer, spectrum, pos, selected_bases[idx], distance):
                            #might be redundant
                            copy_kmer(aux_kmer, original_read, pos, pos + kmer_len)
                            aux_kmer[0] = selected_bases[idx]
                            candidate = transform_to_key(aux_kmer)
                            aux_occurence = give_kmer_multiplicity(spectrum, candidate)
                            if aux_occurence > best_base_occurence:
                                best_base_occurence = aux_occurence
                                best_base = selected_bases[idx]

                    if best_base > 0 and best_base_occurence > 0:
                        original_read[pos] = best_base
                        done = True
                #checking corrections that have done
                if done:
                    region_indices[region][0] = pos
                    if last_position < 0:
                            last_position = pos
                    if last_position - pos < kmer_len:
                        num_corrections += 1

                        #revert kmer back if corrections done exceeds max_corrections
                        if num_corrections > max_corrections:
                            for base_idx in range(pos, last_position + 1):
                                original_read[base_idx] = local_read[base_idx]

                            region_indices[region][0] = last_position + 1
                            break
                    else:
                        last_position = pos
                        num_corrections = 0
                    continue

                #the correction for the current base is done == False
                break
            #endfor lkmer_idx to 0

    #endfor regions_count

