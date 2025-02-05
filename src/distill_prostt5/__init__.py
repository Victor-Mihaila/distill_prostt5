#!/usr/bin/env python3

import click
import os
import torch
import numpy as np
import glob
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
#from MPROSTT5_bert import MPROSTT5, CustomTokenizer  # Import the mini ProstT5 model
import h5py
from Bio import SeqIO
from loguru import logger

from distill_prostt5.classes.MPROSTT5_bert import MPROSTT5, CustomTokenizer
from distill_prostt5.classes.datasets import ProteinDataset, PrecomputedProteinDataset


log_fmt = (
    "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] <level>{level: <8}</level> | "
    "<level>{message}</level>"
)




# @click.group()
# def cli():
#     "Model Distillation Script for Mini ProstT5 Embedder"
#     pass

# @click.command()
# @click.option('-i', '--input', type=click.Path(exists=True), required=True, help='Path to the input protein AA sequences (FASTA format).')
# @click.option('-t', '--threedi', type=click.Path(exists=True), required=True, help='Path to the input protein 3Di sequences (FASTA format).')
# @click.option('-o', '--outpth', type=click.Path(), required=True, help='Path to save model output.')
# @click.option('--precompute_path', type=click.Path(), required=True, help='Path to save or read precomputed embeddings.')
# def precompute(input, threedi, outpth, precompute_path):
#     "Precompute embeddings."
#     click.echo(f"Precomputing embeddings from {input} and {threedi}, saving to {precompute_path}.")
#     # Add precompute logic here

# @click.command()
# @click.option('--model_ckpt', type=click.Path(exists=True), required=True, help='Path to a pre-trained model checkpoint.')
# @click.option('--cnn_checkpoint', type=click.Path(exists=True), help='Path to CNN checkpoint.')
# @click.option('-o', '--outpth', type=click.Path(), required=True, help='Path to save merged output.')
# def merge(model_ckpt, cnn_checkpoint, outpth):
#     "Merge precomputed embeddings with model checkpoints."
#     click.echo(f"Merging using model checkpoint {model_ckpt} and CNN checkpoint {cnn_checkpoint}, saving to {outpth}.")
#     # Add merging logic here

# @click.command()
# @click.option('-i', '--input', type=click.Path(exists=True), required=True, help='Path to the input protein AA sequences (FASTA format).')
# @click.option('-t', '--threedi', type=click.Path(exists=True), required=True, help='Path to the input protein 3Di sequences (FASTA format).')
# @click.option('--model_ckpt', type=click.Path(exists=True), help='Path to a pre-trained model checkpoint (optional).')
# @click.option('-o', '--outpth', type=click.Path(), required=True, help='Path to save model output.')
# def train(input, threedi, model_ckpt, outpth):
#     "Train the model."
#     click.echo(f"Training model with input {input} and {threedi}, checkpoint {model_ckpt}, saving output to {outpth}.")
#     # Add training logic here

# cli.add_command(precompute)
# cli.add_command(merge)
# cli.add_command(train)



@click.group()
@click.help_option("--help", "-h")
def main_cli():
    "Model Distillation Scripts for Mini ProstT5 Model"
    pass



"""
precompute command
"""


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-i",
    "--input",
    help="Path to protein amino acid input file in FASTA format",
    type=click.Path(),
    required=True,
)
@click.option(
    "-p",
    "--precompute_path",
    help="Path to output file where you want to save hdf5 embeddings and other data required for the distillation. Use suffix .h5 (for use with merge)",
    type=click.Path(),
    required=True,
)
@click.option(
    "-m",
    "--max_length",
    help="Max length of input (sequences above this length will be truncated to this many characters).",
    type=int,
    default=768,
)
def precompute(
    ctx,
    input,
    precompute_path,
    max_length,
    **kwargs,
):
    """precomputes ProstT5 embeddings for distillation and tokenises input"""


    logger.info("Beginning precomputation of embeddings")

    # Loading the BERT Tokenizer
    bert_tokenizer = CustomTokenizer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Parse FASTA files
    aa_records = {record.id: str(record.seq) for record in SeqIO.parse(input, "fasta")}
    logger.info(f"Loaded {len(aa_records)} sequences from {input}")
    
    # Load ProstT5 model - needed for embedding generation
    prost_model_name = "Rostlab/ProstT5"
    prost_tokenizer = T5Tokenizer.from_pretrained(prost_model_name)
    prost_model = T5EncoderModel.from_pretrained(prost_model_name).eval().to(device)


    logger.info(f"Starting Computing Mini ProstT5 embeddings for {len(aa_records)} sequences from {input}")

    # reead in the ProstT5 CNN
    repo_root = Path(__file__).parent.resolve()
    CNN_DIR = repo_root / "cnn/"    
    cnn_checkpoint_path = Path(CNN_DIR) / "cnn_chkpnt" / "model.pt"

    train_set = ProteinDataset(aa_records, prost_model, prost_tokenizer, bert_tokenizer, cnn_checkpoint_path, max_length)
    train_set.process_and_save(precompute_path) # dataset.h5

    logger.info(f"Finished Computing Mini ProstT5 embeddings for {len(aa_records)} sequences from {input}")
    logger.info(f"Saved to {precompute_path}")


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-d",
    "--directory",
    help="Directory containing hdf5 files created with distill_prostt5 precompute. Suffix MUST be .h5 for all. Will automatically detect and merge all.",
    type=click.Path(),
    required=True,
)
@click.option(
    "-p",
    "--precompute_path",
    help="Path to output file where you want to save combined hdf5 embeddings and other data required for the distillation",
    type=click.Path(),
    required=True,
)
def merge(
    ctx,
    directory,
    precompute_path,
    **kwargs,
):
    """merges precomputes embeddings and tokenised input for distillation"""

    logger.info(f"Finding all .h5 files in {directory} to merge")
    file_paths = glob.glob(os.path.join(directory, '**', '*.h5'), recursive=True)

    logger.info(f"Found {len(file_paths)} .h5 files in {directory}")
    logger.info(f"There are {file_paths}")
    logger.info(f"Starting merging into {precompute_path}")


    with h5py.File(precompute_path, "w") as merged_file:
        current_index = 0
        
        # Iterate over each HDF5 file
        for file_path in file_paths:
            with h5py.File(file_path, "r") as f:
                # Iterate over the groups in the current file
                for i, group_name in enumerate(f.keys()):
                    group = f[group_name]
                    new_group_name = str(current_index + i)
                    new_group = merged_file.create_group(new_group_name)
                    
                    # Copy datasets from the current group
                    for name, data in group.items():
                        new_group.create_dataset(name, data=data[()])
                
                # Update the index for the next file's groups
                current_index += len(f.keys())

    logger.info(f"Finished merging into {precompute_path}")
    


@main_cli.command()
@click.help_option("--help", "-h")
@click.pass_context
@click.option(
    "-p",
    "--train_path",
    help="Path to .h5 file containing training data processed with the precompute subcommand ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-e",
    "--eval_path",
    help="Path to .h5 file containing evaluation data processed with the precompute subcommand ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Output directory where checkpoints will be saved ",
    type=click.Path(),
    required=True,
)
@click.option(
    "-m",
    "--model_ckpt",
    help="Model checkpoint directory (to restart training from here) ",
    type=click.Path()
)
@click.option(
    "-b",
    "--batch_size",
    help="Batch size",
    type=int,
    default=16
)
@click.option(
    "-e",
    "--epochs",
    help="Epochs",
    type=int,
    default=50
)
def train(
    ctx,
    train_path,
    eval_path,
    output_dir,
    model_ckpt,
    batch_size,
    epochs,
    **kwargs,
):
    """Trains distilled Mini ProstT5 model"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get training dataset
    train_set = PrecomputedProteinDataset(train_path)  # dataset.h5
    eval_set = PrecomputedProteinDataset(eval_path)  # dataset.h5

    # Initialize Mini ProstT5 Model
    model = MPROSTT5().to(device)
    # Print number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    total_params = sum(p.numel() for p in model_parameters)
    logger.info(f"Mini ProstT5 Total Trainable Parameters: {total_params}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="steps",
        logging_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=25,
        save_steps=1000,     
        logging_steps=25,
        learning_rate=3e-4,
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size, # batch size
        gradient_accumulation_steps=1,
        num_train_epochs=500,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set
    )

    # Train the model
    if model_ckpt:
        trainer.train(resume_from_checkpoint=model_ckpt)
    else:
        trainer.train()
    




def main():
    main_cli()


if __name__ == "__main__":
    main()
