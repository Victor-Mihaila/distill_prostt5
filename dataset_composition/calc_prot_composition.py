from Bio import SeqIO
from collections import Counter
import sys

def calculate_aa_composition(fasta_path):
    i=0

    total_counts = Counter()
    total_length = 0
    
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq).upper()
        counts = Counter(seq)
        total_counts.update(counts)
        total_length += len(seq)
        i+=1
        print(i)
    # Only consider standard amino acids
    standard_aas = 'ACDEFGHIKLMNPQRSTVWY'
    aa_counts = {aa: total_counts.get(aa, 0) for aa in standard_aas}
    aa_freq = {aa: count / total_length for aa, count in aa_counts.items()}

    print("Total amino acid composition:")
    for aa in sorted(aa_freq.keys()):
        print(f"{aa}: {aa_freq[aa]:.4f}", end=' ')
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aa_composition.py protein_sequences.fasta")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    calculate_aa_composition(fasta_file)


