# TODOS

- The program should be able to accept configs or environment variables

- Distribute the reads near even to the available GPU for a more faster execution time of the multistage on the whole dataset

- Produce accuracy results from our correction method.



 
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

- Deez nuts
