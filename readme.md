# TODOS

- Store dataframes within the actor in order for efficient utilization of downstream processes
- Compare the result and quality of correction by utilizing varying numbers of GPUs
- Organize driver code into OOP 
- Create bash script in order to automate the process of running the program
- During kmer extraction, clean the reads or remove the N bases in order to avoid deeming N kmers as valid kmers
- During transformation into 1d read, do not remove N bases, put them as is since they wont be corrected since they are not in the kmer spectrum
# PROBLEM
- How can I efficiently extract kmers? Is it okay to utilize all available GPUs? 

# POSSIBLE OPTIMIZATIONS WE CAN DO
- Use rayon within rust in order to parallelize reading of reads and writing corrected reads
- Utilize all available GPUs in order to parallelize the correction process


