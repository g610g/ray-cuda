# TODOS

- The program should be able to accept configs or environment variables
- Create a bash script to capture env variables and run the python parallel script
- Distribute the reads near even to the available GPU for a more faster execution time of the multistage on the whole dataset

- Produce accuracy results from our correction method. (resolved)

- Complete first all of the error correction algorithm (partially done)

- We will stick with using small length of kmer as of the moment and try to force if we can get a good gain value. result: (its hard to push low kmer len, minimum length should be 13 for kmer to work properly)

- GPU threads are accessing elements on a non coalesced way plus the problem of thread divergence. We should try to achieve this with minor changes on the kernel implementation 

- Querying the spectrum is a bottleneck since on Musket, they can use hash table which takes O(1) querying as compared to ours where we can only have a max of O(logn) querying optimization.(Partially solved)
-Create unit tests for one sided correction in order to validate implementation and logics

 
# FINISHED

- Core implementation of one sided and two sided looks good
- The two multistages does corrects and mutates the reads
- Implemented on a single GPU
- Voting based refinement implementation
- Solved the problem of long query for kmer spectrum by sorting the kmer spectrum and use binary search for searching

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

