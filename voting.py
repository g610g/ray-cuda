# something is wrong with the voting algo the changes falsely correct the reads
from numba import cuda
from helpers import transform_to_key, in_spectrum


@cuda.jit(device=True)
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


# check for indexing in the code snippet
@cuda.jit(device=True)
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


# the voting refinement
@cuda.jit
def voting_algo(dev_reads, offsets, kmer_spectrum, kmer_len):
    threadIdx = cuda.grid(1)

    if threadIdx < offsets.shape[0]:

        MAX_LEN = 150
        start, end = offsets[threadIdx][0], offsets[threadIdx][1]

        vm = cuda.local.array((4, MAX_LEN), "uint8")
        bases = cuda.local.array(4, "uint8")

        for idx in range(4):
            bases[idx] = idx + 1

        for idx in range(start, end - (kmer_len - 1)):
            curr_kmer = transform_to_key(dev_reads[idx : idx + kmer_len], kmer_len)

            # invoke voting if the kmer is not in spectrum
            if not in_spectrum(kmer_spectrum, curr_kmer):
                invoke_voting(vm, kmer_spectrum, bases, kmer_len, idx, dev_reads, start)

        # apply the result of the vm into the reads
        apply_vm_result(vm, dev_reads, start, end)
