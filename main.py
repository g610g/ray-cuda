size = 3012334
cpus = 48
batch_size = size // cpus


idx = [(batch_idx // batch_size) for batch_idx in range(0, size, batch_size)]
# print(idx)
for i in range(9, 9 + 1):
    print("Hello world")
