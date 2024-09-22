import numpy as np
from typing import List, Tuple, NewType, Union
import torch
from torch import Tensor
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol
from tdc import Oracle
FlatRewards = NewType("FlatRewards", Tensor)
from gflownet.utils.common import set_device, set_float_precision

class drd2():
    """
    Building Block oracle
    """
        
    def __init__(self,device,float_precision,higher_is_better=None,**kwargs):
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.higher_is_better = higher_is_better
        self.model = Oracle(name='DRD2')
        
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
    
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float))

    def __call__(self, states: List) -> Tuple[FlatRewards, Tensor]:
        smis_valid = []
        pos = []
        preds = [0.0] * len(states)
        for i,smi in enumerate(states):
            if smi is not None:
                mol=Chem.MolFromSmiles(smi)
                if mol:
                    smis_valid.append(smi)
                    pos.append(i)
        pred = self.model(smis_valid)
        for pos, value in zip(pos, pred):
            preds[pos] = value
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds).view(-1)