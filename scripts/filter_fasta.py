from Bio import SeqIO
from argparse import RawTextHelpFormatter
import argparse

def get_input():
    """gets input for decoding prostt5 dataset
    :return: args
    """
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--cluster",
        action="store",
        required=True,
        help="cluster validation FASTA",
    )
    parser.add_argument(
        "-i",
        "--input",
        action="store",
        required=True,
        help="larger input FASTA",
    )
    parser.add_argument(
        "-o", "--output", action="store", default="", 
        help="output FASTA."
    )

    args = parser.parse_args()
    return args

def filter(cluster, input, output):

    ids_to_filter = set(record.id for record in SeqIO.parse(cluster, "fasta"))

    # Filter and write records from the second FASTA
    with open(output, "w") as out_handle:
        for record in SeqIO.parse(input, "fasta"):
            if record.id not in ids_to_filter:
                SeqIO.write(record, out_handle, "fasta")


if __name__ == "__main__":
    args = get_input()
    filter(args.cluster, args.input, args.output)

