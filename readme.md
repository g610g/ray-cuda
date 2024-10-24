# TODOS

- Tracking of the number of corrections made for each kmer and rolling back each corrections made in a kmer if the corrections made is > max number of corrections
- The program should be able to accept configs or environment variables
- Two sided and one sided should be done on an interative approach
- Distribute the reads near even to the available GPU for a more faster execution time of the multistage on the whole dataset
- 
# FINISHED

- Core implementation of one sided and two sided looks good
- The two multistages does corrects and mutates the reads
- Implemented on a single GPU
