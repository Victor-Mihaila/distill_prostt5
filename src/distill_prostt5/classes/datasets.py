
"""
Define the dataset
"""

import torch.nn as nn
import torch
import h5py
from .CNN import CNN
from torch.utils.data import Dataset
from tqdm import tqdm

class ProteinDataset(Dataset):
    def __init__(self, aa_records, prost_model, prost_tokenizer,bert_tokenizer, cnn_checkpoint, max_length):
        self.aa_records = aa_records
        self.prost_model = prost_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.prost_tokenizer = prost_tokenizer
        self.bert_tokenizer = bert_tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cnn_checkpoint = cnn_checkpoint
        self.max_length = max_length
        self.data = []

    def __len__(self):
        return len(self.aa_records)  # Return the number of items in your dataset

    def process_and_save(self, save_path):
        with h5py.File(save_path, "w") as h5f:
            i = 0 
            #for key  in self.aa_records.keys():
            for key in tqdm(self.aa_records.keys(), desc="Processing sequences"):

                # generate tokens for ProstT5 embedding generation
                aa_seq = self.aa_records[key]
                prostt5_prefix = "<AA2fold>"
                aa_seq_pref = prostt5_prefix + " " + " ".join(aa_seq)
                prost_tokens = self.prost_tokenizer(aa_seq_pref, return_tensors="pt", padding='max_length', truncation=True,  max_length=self.max_length+1) # max_length +1 to let us strip off the prostt5 prefix and keep the same size
                prost_input_ids = prost_tokens.input_ids.to(self.device)
                prost_attention_mask = prost_tokens.attention_mask.to(self.device)

                # print("prostT5 tokens")
                # print(prost_input_ids)
                # print(prost_input_ids.shape)

                # generate tokens for mini ProstT5 (bert0)

                bert_tokens = self.bert_tokenizer(aa_seq, return_tensors="pt", padding='max_length', truncation=True,  max_length=self.max_length)
                # need the labels for the padding mask later for loss calc
                # probably a better way to do this but whatever it works for now
                bert_labels = bert_tokens.input_ids.to(self.device) # these will also be fine as the pad is 0
                bert_input_ids = bert_tokens.input_ids.to(self.device)
                bert_attention_mask = bert_tokens.attention_mask.to(self.device)

                # print("bert tokens")
                # print(bert_input_ids)
                # print(bert_input_ids.shape)

                # to generate the ProstT5 logits

                with torch.no_grad():
                    # follows translate.py/phold
                    residue_embedding = self.prost_model.encoder(prost_input_ids, attention_mask=prost_attention_mask).last_hidden_state
                    residue_embedding = ( # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
                        residue_embedding
                        *prost_attention_mask.unsqueeze(dim=-1)
                    )
                    residue_embedding = residue_embedding[:, 1:] # strip off the AA2fold token
                    predictor = CNN().to(self.device)
                    state = torch.load(self.cnn_checkpoint, map_location=self.device)
                    predictor.load_state_dict(state["state_dict"])
                    prediction = predictor(residue_embedding.to(self.device))
                    logits = prediction.transpose(1, 2) 

                # Save tensors to HDF5
                grp = h5f.create_group(str(i))
                grp.create_dataset("input_ids", data=bert_input_ids.cpu().numpy())
                grp.create_dataset("labels", data=bert_labels.cpu().numpy())
                grp.create_dataset("attention_mask", data=bert_attention_mask.cpu().numpy())
                grp.create_dataset("target", data=logits.cpu().numpy())
                i += 1

        print(f"Dataset saved to {save_path}")


"""
Define reading dataset once precomputed
"""

class PrecomputedProteinDataset(Dataset):
    def __init__(self, hdf5_path):
        self.hdf5_path = hdf5_path
        self.h5f = h5py.File(self.hdf5_path, "r")

    def __len__(self):
        return len(self.h5f.keys())

    def __getitem__(self, idx):
        
        grp = self.h5f[str(idx)]
        input_ids = torch.tensor(grp["input_ids"][:], dtype=torch.long)
        labels = torch.tensor(grp["labels"][:], dtype=torch.long)
        attention_mask = torch.tensor(grp["attention_mask"][:], dtype=torch.long)
        target = torch.tensor(grp["target"][:], dtype=torch.float)

        return {
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "target": target.squeeze(0)
        }

