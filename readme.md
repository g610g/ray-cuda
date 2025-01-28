# TODOS

- Voting base refinement has no effect at the result, I should examine it properly if my understanding is sakto and generate a proper test cases to validate nga sakto ang akoang code for it.
- Check the solids array produced after doing two sided correction

- Verify implementation of successor if no bugs (1)

- Create predeccessor implementation (going to the left of the region)




 
# PROBLEM
- How to effieciently transform back jagged array into a list of reads

- How can we do it by leveraging the power of cudf?  (resolved but another problem comes with serialization of return data from tasks since data is not in numpy format)

- Still havent transformed corrected reads into a Bio.Seq.Seq class and then putting it back to its corresponding SeqRecord object (resolved)

- We are slower than the original musket if a lot of threads will be used for musket

- When kmer length is 13, the cutoff threshold is low, it also means that when the cutoff threshold is low, the number of kmers during filtering is very high. It takes about 7mil kmers are still there for 13mers, and when we try to use the in_spectrum device function, it takes a lot of time. (resolved by utilizing binary search)

# POSSIBLE OPTIMIZATIONS WE CAN DO
- Run the multistage on the number of available GPUs in the system by distributing even amount of reads into each GPU
- The process of translating each reads from Seq.Seq objects into strings can be done by utilizing all available CPU cores
- Refactor the GPUExtractor class 
- Remove the variables within the kernels that are used for benchmarking



# TESTING PROBLEM
- During lookahead validation, bases at each end of the read doesnt have any neighbor
- Since two sided cannot correct bases that are at the leftmost and rightmost, it fails to correct those erroneous bases. We try to let one sided handle that case. If ang solids array is something like [-1, 1, 1, -1, -1, -1] with a kmer length of 3, ang mga last 3 bases, dili na sha ma correct. Probably ang first base is pwede sha ma correct because sa kana nga kmer, naa lay isa ka sequencing error. 


# ONE SIDED EDGE CASES
- During sa pag one sided correction. If I revert ang bases, ang regions indices from that corrections is need bapod  i reset?
- If prev region end is nag correct and gi revert, then for the next region executing correction with orientation to the left is  ma apektohan 
