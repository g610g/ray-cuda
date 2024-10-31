# TODOS

- The program should be able to accept configs or environment variables

- Distribute the reads near even to the available GPU for a more faster execution time of the multistage on the whole dataset

- Produce accuracy results from our correction method.

- Complete first all of the error correction algorithm

- We will stick with using small length of kmer as of the moment and try to force if we can get a good gain value

- The purpose of using small length of kmer lies to the fact that gpu kernel gets slow when having a long len of kmer since base on my assumption lies on the memory access

- GPU threads are accessing elements on a non coalesced way plus the problem of thread divergence


 
# FINISHED

- Core implementation of one sided and two sided looks good
- The two multistages does corrects and mutates the reads
- Implemented on a single GPU
- Voting based refinement implementation

# PROBLEM
- How to effieciently transform back jagged array into a list of reads

- How can we do it by leveraging the power of cudf.

- How to create a dataframe where each row is the array from jagged array created by referencing from the offsets 2d array

- Still havent transformed corrected reads into a Bio.Seq.Seq class and then putting it back to its corresponding SeqRecord object

- We are slower than the original musket if a lot of threads will be used for musket

# POSSIBLE OPTIMIZATIONS WE CAN DO
- Run the multistage on the number of available GPUs in the system by distributing even amount of reads into each GPU
- The process of translating each reads from Seq.Seq objects into strings can be done by utilizing all available CPU cores
- Refactor the GPUExtractor class 
- Remove the variables within the kernels that are used for benchmarking

