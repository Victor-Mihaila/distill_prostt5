#!/usr/bin/env python3

import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter protein FASTA to keep sequences longer than a given length"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input protein FASTA file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output FASTA file"
    )
    parser.add_argument(
        "-l", "--min-length",
        type=int,
        default=512,
        help="Minimum sequence length (default: 512)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    kept = 0
    total = 0

    with open(args.output, "w") as out_f:
        for record in SeqIO.parse(args.input, "fasta"):
            total += 1
            if len(record.seq) > args.min_length:
                SeqIO.write(record, out_f, "fasta")
                kept += 1

    print(f"Processed: {total}")
    print(f"Kept (> {args.min_length} aa): {kept}")


if __name__ == "__main__":
    main()
