from test_modules import in_spectrum, transform_to_key



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
