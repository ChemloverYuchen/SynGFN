import numpy as np
from typing import List, Tuple, NewType, Union
from torchtyping import TensorType
import torch
from torch import Tensor
import torch_geometric.data as gd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol as RDMol

from gflownet.gflownet.utils import bengio2021flow
FlatRewards = NewType("FlatRewards", Tensor)

from gflownet.gflownet.utils.common import set_device, set_float_precision

class SEH:
    """
    Building Block oracle
    """
        
    def __init__(
        self,
        device,
        cost, 
        maximize,
        float_precision,
        fid,
        higher_is_better=None,
        **kwargs
        ):
        self.cost = cost
        self.maximize = maximize
        self.fid = fid
        self.device = set_device(device)
        self.float = set_float_precision(float_precision)
        self.higher_is_better = higher_is_better
        self.model = self._load_task_model().to(self.device)
        
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
    
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float) / 8)

    def _load_task_model(self):
        model = bengio2021flow.load_original_model()
        return model
    
    def __call__(self, states: List) -> Tuple[FlatRewards, Tensor]:
        mols = [Chem.MolFromSmiles(s) if s else None for s in states]
        graphs = [bengio2021flow.mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 1))), is_valid
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        batch.to(self.device)
        preds = self.model(batch).reshape((-1,)).data.to(self.device)
        preds[preds.isnan()] = 0
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds).view(-1)