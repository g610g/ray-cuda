from test_modules import in_spectrum, transform_to_key, copy_kmer

def cast_votes(local_read, vm, seq_len, kmer_len, bases, size, kmer_spectrum, ascii_kmer):
    #reset voting matrix
    for i in range(seq_len):
        for j in range(len(bases)):
            vm[i][j] = 0
    max_vote = 0
    #check each kmer within the read (planning to put the checking of max vote within this iteration)
    for ipos in range(0, size + 1):
        
        copy_kmer(ascii_kmer, local_read, ipos, ipos + kmer_len)
        kmer = transform_to_key(ascii_kmer, kmer_len)
        if in_spectrum(kmer_spectrum, kmer):
            continue
        for base_idx in range(kmer_len):
            original_base = ascii_kmer[base_idx]
            for base in bases:
                if original_base == base:
                    continue
                ascii_kmer[base_idx] = base
                candidate = transform_to_key(ascii_kmer, kmer_len)
                if in_spectrum(kmer_spectrum, candidate):
                    vm[ipos + base_idx][base - 1] += 1
            ascii_kmer[base_idx] = original_base

    #find maximum vote
    max_vote_idx = -1
    for ipos in range(0, seq_len):
        for idx in range(len(bases)):
           if vm[ipos][idx] >= max_vote:
                max_vote = vm[ipos][idx]
                max_vote_idx = ipos

    print(f"Voting matrix result: {vm[max_vote_idx]}")
    return (max_vote, max_vote_idx)
def apply_voting_result(local_read, vm, seq_len, bases, max_vote):
    for ipos in range(seq_len):
        alternative_base = -1
        for base_idx in range(len(bases)):
            if vm[ipos][base_idx] == max_vote:
                if alternative_base == -1:
                    alternative_base = base_idx + 1
                else:
                    alternative_base = -1

        #apply the base correction if we have found an alternative base
        if alternative_base >= 1:
            local_read[ipos] = alternative_base

def invoke_voting(vm, kmer_spectrum, bases, kmer_len, curr_idx, read, start):

    curr_kmer = read[curr_idx : curr_idx + kmer_len]
    for idx in range(kmer_len):
        original_base = curr_kmer[idx]
        for base in bases:
            curr_kmer[idx] = base
            trans_curr_kmer = transform_to_key(curr_kmer, kmer_len)

            # add a vote to the corresponding index
            if in_spectrum(kmer_spectrum, trans_curr_kmer):
                vm[base - 1][(curr_idx - start) + idx] += 1

        # revert the base back to its original base
        curr_kmer[idx] = original_base

def apply_vm_result(vm, read, start, end):
    # foreach base position, we check if there are bases that have values greater than 0
    # apply that base if its value is greater than 0, otherwise, just ignore and retain the original base
    for read_position in range(end - start):
        current_base = -1
        max_vote = 0
        for base in range(4):
            if vm[base][read_position] > max_vote:
                current_base = base + 1
                max_vote = vm[base][read_position]
        if max_vote != 0 and current_base != -1:
            read[start + read_position] = current_base
