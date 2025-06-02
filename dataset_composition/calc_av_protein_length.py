from Bio import SeqIO
import sys

def calculate_av_length_composition(fasta_file):

    total_length = 0
    sequence_count = 0

    for record in SeqIO.parse(fasta_file, "fasta"):
        total_length += len(record.seq)
        sequence_count += 1
        print(sequence_count)
    if sequence_count == 0:
        print("No sequences found.")
    else:
        average_length = total_length / sequence_count
        print(f"Average sequence length: {average_length:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calc_av_protein_length.py protein_sequences.fasta")
        sys.exit(1)

    fasta_file = sys.argv[1]
    calculate_av_length_composition(fasta_file)
