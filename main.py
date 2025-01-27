import numpy as np

size = 3012334
cpus = 48
batch_size = size // cpus


arr = [2, 3]
[two, three] = arr
two += 1
print(arr)
print(two)
# idx = [(batch_idx // batch_size) for batch_idx in range(0, size, batch_size)]
# kmer_len = 5
# reads = np.arange(14)
# kmer_counter_list = np.zeros((len(reads) - (kmer_len - 1)), dtype="uint8")
for idx in range(0, -1):
    print(idx)
