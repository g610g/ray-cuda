import numpy as np
import fastq_parser as fp



# Read fastq file
data = np.arange(50, 70, dtype='uint8').reshape(4, 5)
fastq = fp.write_fastq_file('test.fastq', data)
