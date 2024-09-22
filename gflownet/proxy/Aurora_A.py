from collections import OrderedDict
from logging import Logger
import math
import os
import random
from typing import List, Tuple, NewType, Union
import numpy as np
import torch
from torch import Tensor
FlatRewards = NewType("FlatRewards", Tensor)
from rdkit import Chem, RDLogger
from tqdm import tqdm
import pandas as pd
from rdkit.Chem import Descriptors
from chemprop.train import predict
from chemprop.data import MoleculeDataset, MoleculeDataLoader, MoleculeDatapoint
from chemprop.data.utils import get_data, filter_invalid_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
RDLogger.DisableLog('rdApp.*')
from gflownet.utils.common import set_device, set_float_precision
tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))

def get_data_from_smiles(smiles: List[List[str]],
                         skip_invalid_smiles: bool = True,
                         logger: Logger = None,
                         features_generator: List[str] = None) -> MoleculeDataset:
    """
    Converts a list of SMILES to a :class:`~chemprop.data.MoleculeDataset`.

    :param smiles: A list of lists of SMILES with length depending on the number of molecules.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles using :func:`filter_invalid_smiles`
    :param logger: A logger for recording output.
    :param features_generator: List of features generators.
    :return: A :class:`~chemprop.data.MoleculeDataset` with all of the provided SMILES.
    """
    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([
        MoleculeDatapoint(
            smiles=[smile], 
            row=OrderedDict({'smiles': smile}),
            features_generator=features_generator
        ) for smile in smiles
    ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data

class chemprop_model():
    def __init__(self, device, float_precision, checkpoint_dir='gflownet/proxy/QSAR/Aurora_A/model.pt', higher_is_better=None):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.higher_is_better = higher_is_better
        checkpoint_file = checkpoint_dir
        self.scaler, self.features_scaler, self.atom_descriptor_scaler, self.bond_feature_scaler, _= load_scalers(checkpoint_file)
        self.train_args = load_args(checkpoint_file)
        self.model = load_checkpoint(checkpoint_file, self.device)
        
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
        
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float))  
    
    def __call__(self, smiles, batch_size=32):
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
        valid_indices = [i for i in range(len(test_data)) if test_data[i].mol != [None]]
        full_data = test_data
        test_data = MoleculeDataset([test_data[i] for i in valid_indices])

        # Normalize features
        if self.train_args.features_scaling:
            test_data.normalize_features(self.features_scaler)
        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
            batch_size=self.train_args.batch_size,
            num_workers=self.train_args.num_workers
        )
        model_preds = predict(
            model=self.model,
            data_loader=test_data_loader,
            scaler=self.scaler
        )
        preds = np.array(model_preds).squeeze(-1).tolist()
        
        # Put zero for invalid smiles
        full_preds = [0.0] * len(full_data)
        #min: 6, max=10
        for i, si in enumerate(valid_indices):
            full_preds[si] = max(0, (preds[i] - 6) / 4)
        full_preds = self.flat_reward_transform(full_preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(full_preds).view(-1)
 