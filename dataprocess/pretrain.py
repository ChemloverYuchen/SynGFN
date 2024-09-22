import argparse
import torch
import torch.nn as nn
import numpy as np
import json
import gzip
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from template import smarts_list
from multiprocessing import Pool
mol_dict={}
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split


def label2onehot(react_key, label):
    label_list = react_key + ["EOS"] + ["UNI"]
    label_to_index = {tag: i for (i,tag) in enumerate(label_list)}
    label_onehot = np.zeros(len(label_list), dtype=np.float32)
    index = label_to_index[label]
    label_onehot[index] = 1.0
    return label_onehot 

def get_patt(smart):
    try:
        mol = mol_dict[smart]
        return mol
    except:
        mol = Chem.MolFromSmarts(smart)
        mol_dict[smart] = mol
        return mol

def statebatch2policy_2(smiles_batch, tags_batch, react_key):
    batch_inputs = []

    for smiles, tag in zip(smiles_batch, tags_batch):
        state_policy = state2policy_2(smiles, tag, react_key)
        batch_inputs.append(state_policy)

    batch_inputs = np.array(batch_inputs, dtype=np.float32)
    
    return batch_inputs

def state2policy_2(smiles,tag,react_key):
    mol = Chem.MolFromSmiles(smiles)
    if tag == "ANY":
        label_onehot = np.zeros(len(label_list), dtype=np.float32)
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)
        features_vec = np.array(features_vec, dtype=np.float32)
        state_policy = np.hstack((features_vec, label_onehot)).astype(np.float32)
    else:
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)
        features_vec = np.array(features_vec, dtype=np.float32)
        label_onehot = label2onehot(react_key,tag)
        state_policy = np.hstack((features_vec, label_onehot)).astype(np.float32)
    return state_policy

class PretrainDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tag = self.dataframe.iloc[idx]['tag']
        smiles = self.dataframe.iloc[idx]['smiles']
        sample = {'tag': tag, 'smiles': smiles}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data_and_split():
    with gzip.GzipFile(args.mask_dict_path, 'r') as f_in:
        mask_dict = json.load(f_in)
    react_key = list(mask_dict.keys())
    label_list = react_key + ["EOS"] + ["UNI"]

    with gzip.GzipFile(args.smiles_list_path, 'r') as f_in:
        smiles_list = json.load(f_in)
    
    mask_dict_new = mask_dict.copy()
    mask_dict_new["EOS"] = [len(smiles_list)]
    mask_dict_new["UNI"] = [len(smiles_list), len(smiles_list) + 1]
    mask_dict_new["ANY"] = list(range(len(smiles_list)))

    vocab_1 = smarts_list + ["EOS"]
    inverse_lookup_1 = {i: a for (i, a) in enumerate(vocab_1)}
    mol_dict = {}

    df = pd.read_csv(args.pretrain_data_path)

    def sample_by_tag(group):
        return group.sample(min(args.train_num, len(group)), random_state=42)

    df = df.groupby("tag", group_keys=False, as_index=False).apply(sample_by_tag)
    df.reset_index(drop=True, inplace=True)
    
    train_df, test_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df['tag'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['tag'])

    return {
        "train": PretrainDataset(train_df),
        "val": PretrainDataset(val_df),
        "test": PretrainDataset(test_df),
    }, label_list, smiles_list, mask_dict_new, react_key

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        n_layers = 2
        layers_dim = [input_size] + [hidden_size] * n_layers + [output_size]
        self.layers = nn.Sequential(
            *(sum([[nn.Linear(idim, odim)] + ([nn.LeakyReLU()] if n < len(layers_dim) - 2 else []) for n, (idim, odim) in enumerate(zip(layers_dim, layers_dim[1:]))], []))
        )

    def forward(self, x):
        return self.layers(x)

class ValidReactLossWithMargin(nn.Module):
    def __init__(self, margin=3.0):
        super(ValidReactLossWithMargin, self).__init__()
        self.margin = margin
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, tags, mask_dict_new):
        batch_loss = 0.0
        for i, tag in enumerate(tags):
            valid_indices = torch.zeros_like(outputs[i])
            valid_indices[mask_dict_new[tag]] = 1
            invalid_indices = 1 - valid_indices

            loss = self.bce_loss(outputs[i], valid_indices)

            valid_outputs = outputs[i] * valid_indices
            invalid_outputs = outputs[i] * invalid_indices
            margin_loss = torch.clamp(self.margin - valid_outputs + invalid_outputs, min=0)
            loss += margin_loss.mean()

            batch_loss += loss

        return batch_loss / len(tags)
    
def save_model_state_dict(model, file_path):
    state_dict = model.state_dict()
    new_state_dict = {key.replace('layers.', ''): value for key, value in state_dict.items()}
    torch.save(new_state_dict, file_path)

def train_and_evaluate(model, dataloaders, optimizer, loss_function, mask_dict_new, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloaders['train'], desc=f"Training Epoch {epoch+1}"):
            inputs, labels = batch['smiles'], batch['tag']
            inputs = statebatch2policy_2(inputs, labels, react_key)
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(inputs)
            loss = loss_function(outputs, labels, mask_dict_new)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(dataloaders['train'])}")
        
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for batch in tqdm(dataloaders['val'], desc=f"Validation Epoch {epoch+1}"):
                inputs, labels = batch['smiles'], batch['tag']
                inputs = statebatch2policy_2(inputs, labels, react_key)
                inputs = torch.tensor(inputs, dtype=torch.float32)
                outputs = model(inputs)
                loss = loss_function(outputs, labels, mask_dict_new)
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Validation Loss: {total_loss / len(dataloaders['val'])}")
    
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in tqdm(dataloaders['test'], desc="Testing"):
            inputs, labels = batch['smiles'], batch['tag']
            inputs = statebatch2policy_2(inputs, labels, react_key)
            inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = model(inputs)
            loss = loss_function(outputs, labels, mask_dict_new)
            total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(dataloaders['test'])}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pre-train the policy model 2')
    parser.add_argument('--pretrain_data_path', type=str, default='final/pretrain_mw150.csv', help='Path to the pretrain data file')
    parser.add_argument('--mask_dict_path', type=str, default='final/mask_dict_mw150.json.gz', help='Path to the mask dictionary file')
    parser.add_argument('--smiles_list_path', type=str, default='final/smiles_list_mw150.json.gz', help='Path to the smiles list file')
    parser.add_argument('--model_save_path', type=str, default='pretrain/pretrain_mw150.ckpt', help='Path to the saved model')
    parser.add_argument('--train_num', type=int, default=5, help='train number for each tag')
    parser.add_argument('--n_hid', type=int, default=128, help='The number of neurons in the hidden layer')
    parser.add_argument('--n_epochs', type=int, default=1, help='The number of training epochs')
    args = parser.parse_args()
    datasets, label_list, smiles_list, mask_dict_new, react_key = load_data_and_split()
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=32, shuffle=True, num_workers=32)
        for x in ['train', 'val', 'test']
    }
    input_size = 4096 + len(label_list)
    hidden_size = args.n_hid
    output_size = len(smiles_list) + 2
    model = MLP(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = ValidReactLossWithMargin()
    train_and_evaluate(model, dataloaders, optimizer, loss_function, mask_dict_new, num_epochs=args.n_epochs)
    save_model_state_dict(model, args.model_save_path)