import pandas as pd
from Bio import SeqIO
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def read_fasta_and_process(file_name):
    # Reading the fasta file
    sequences = []
    for record in SeqIO.parse(file_name, "fasta"):
        sequences.append(str(record.seq))

    # Removing the middle element 'T'
    n = len(sequences)
    m = (len(sequences[0]) + 1) // 2
    for i in range(n):
        sequences[i] = sequences[i][:m-1] + sequences[i][m:]  # remove the middle character

    return sequences, n

def compute_ppt(sequences, n):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    PPT = pd.DataFrame(0, index=range(n), columns=list(AA))

    # Fill the PPT matrix
    for idx, seq in enumerate(sequences):
        m = len(seq)
        for char in seq:
            if char in AA:
                PPT.at[idx, char] += 1
        PPT.iloc[idx] = PPT.iloc[idx] / m  # Normalize by the length of sequence

    return PPT

# Process positive samples
positive_sequences, Np = read_fasta_and_process('testmany.txt')
PPT1 = compute_ppt(positive_sequences, Np)
PPT1.dropna(how='all', inplace=True)
PPT1.to_csv('testmany.csv', index=False, header=False)

# Process negative samples
# negative_sequences, Nn = read_fasta_and_process('kcr_INDN.txt')
# PPT2 = compute_ppt(negative_sequences, Nn)
# PPT2.to_csv('AAC_INDN.csv', index=False)