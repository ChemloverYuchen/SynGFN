import numpy as np
from typing import List, Tuple, NewType, Union
from torchtyping import TensorType
import torch
from torch import Tensor
import re
import math
import json
import os
from vina import Vina
import pandas as pd
from functools import partial
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
RDLogger.DisableLog('rdApp.*')
DEBUG=False
import warnings
warnings.filterwarnings("ignore")

seed = 666
FlatRewards = NewType("FlatRewards", Tensor)

from gflownet.gflownet.utils.common import set_device, set_float_precision

root_path = os.path.dirname(os.path.abspath(__file__))
vina_path = os.path.join(root_path, 'vina')

def get_mol(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        Chem.Kekulize(mol)
        Chem.SanitizeMol(mol)
    except:
        return None
    return mol

def smi2pdbqt(smiles):
    d = ['', '1', '2', '3', '4', '5', '6', '7', '8', '9', '>', '<', '[', ']', '(', ')', '#', '-', '=', 'H', 'C', 'c', 'S', 's', 'N', 'n', 'O', 'o', 'F', 'Cl', 'Br', ':', '@','\\', '//', '/']
    reg = re.compile('(Br|Cl|.)')
    atoms = [atom for atom in reg.split(smiles)[1::2] if atom not in d]
    if atoms:
        print(atoms, " is invalid atom.")
        return None
    mol = get_mol(smiles)
    if not mol:
        return None
    try:
        m3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m3d, randomSeed=seed)
        try:
            AllChem.MMFFOptimizeMolecule(m3d)
        except:
            return None
        preparator = MoleculePreparation()
        preparator.prepare(m3d)
        pdbqt_string = preparator.write_pdbqt_string()
        return pdbqt_string
    except:
        return None
    
def _vina_dock(path, dmap, exhaustiveness, n_poses, write_pose, smi):
    ligand = smi2pdbqt(smi)
    if not ligand:
        print(smi,' invalid')
        return 0
    try:
        v = Vina(sf_name='vina',seed=seed, verbosity=0, cpu=10)
        v.load_maps(dmap)
        v.set_ligand_from_string(ligand)
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        if write_pose:
            v.write_poses(f'{path}/vina_out_{smi}.pdbqt', n_poses=5, overwrite=True)
        return v.score()[0]
    except:
        return 0

class dockProxy:
    def __init__(self, dock_root):
        self.dscore_map = {}
        self.dock_root = dock_root
        assert os.path.exists(self.dock_root)
            
    def run_dock(self, dock_grids, smiles, method='vina'):
        
        if type(dock_grids) == str:
            dock_grids = [dock_grids]
        
        if type(smiles) == str:
            smiles = [smiles]
        elif type(smiles) != list:
            smiles = list(smiles)
        
        dockmol = set(smiles)
        d_scores = np.ones((len(dock_grids), len(smiles)))
        for idx, dock_grid in enumerate(dock_grids):
            dmapfile = f"{self.dock_root}/{dock_grid}.json"

            if dock_grid not in self.dscore_map:
                if not os.path.isfile(dmapfile):
                    json.dump({}, open(dmapfile, 'w'), indent=2)
                    self.dscore_map[dock_grid] = {}
                else:
                    self.dscore_map[dock_grid] = json.load(open(dmapfile, 'r'))
                
            _dockmol = [m for m in dockmol if m not in self.dscore_map[dock_grid] ]
            
            if method=='vina':
                _scores = self.vina_dock(dock_grid, _dockmol)
            elif method == 'sch':
                _scores = self.sch_dock(dock_grid, _dockmol)
            
            self.dscore_map[dock_grid].update(dict(_scores))
            
            self.dscore_map[dock_grid].update(json.load(open(dmapfile, 'r')))
            json.dump(self.dscore_map[dock_grid], open(dmapfile, 'w'), indent=2)
            d_scores[idx] = self.get_result(smiles, dock_grid)
        if len(dock_grids) == 1:
            return d_scores[0]
        else:
            return d_scores
        
    def get_result(self, smiles, dock_grid):
        if dock_grid not in self.dscore_map:
            print(dock_grid, ' not initialized')
            return []
        scores = [self.dscore_map[dock_grid][smi] if smi in self.dscore_map[dock_grid] else 0 for smi in smiles]
        return scores
    
    def vina_dock(self, protein, smiles, exhaustiveness=24, n_poses=10, write_pose=False, thread=20):
        if len(smiles) == 0:
            return []
        dmap = self.make_grid(protein)
        _dock = partial(_vina_dock, self.dock_root, dmap, exhaustiveness, n_poses, write_pose)
        worker = Pool(thread)
        score = worker.map(_dock, smiles)
        worker.close()
        worker.join()
        return [(k,v) for k,v in zip(smiles, score) if v < 0]
    
    def make_grid(self, grid_name):
        grid = f'{self.dock_root}/{grid_name}_map/{grid_name}'
        if os.path.exists(f'{self.dock_root}/{grid_name}_map'):
            return grid
        else:
            os.mkdir(f'{self.dock_root}/{grid_name}_map')
        v = Vina(sf_name='vina', verbosity=0, cpu=10)
        v.set_receptor(f'{self.dock_root}/{grid_name}.pdbqt')
        #for sEH target :-12.973, 27.046, -13.643
        v.compute_vina_maps(center=[0, 0, 0], box_size=[15, 15, 15], force_even_voxels=True)
        v.write_maps(map_prefix_filename=grid)
        return grid
            
    def sch_dock(self, dock_grid, smiles):
        if len(smiles) == 0:
            return []
        score = [0]*len(smiles)
        with open(f'{self.dock_root}/ligand.smi', 'w') as file:
            for idx,smi in enumerate(smiles):
                file.write(f'{smi} {idx}\n')
        with open(f'{self.dock_root}/grid.info', 'w') as file:
            file.write(dock_grid)
        a = os.system(f'{self.dock_root}/dock.sh')
        if a ==0:
            df = pd.read_csv(f'{self.dock_root}/output.csv').loc[1:]
            for i, r in df.iterrows():
                score[int(r.NAME)] = r['r_i_docking_score']
        else:
            raise Exception('Schrodinger Docking Wrong')
        return [(k,v) for k,v in zip(smiles, score)]

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
        self.dock = dockProxy(vina_path)
        
    def setup(self, env=None):
        self.max_seq_length = env.max_seq_length
    
    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y,dtype=self.float) / -12)

    def __call__(self, states: List) -> Tuple[FlatRewards, Tensor]:
        preds = self.dock.run_dock('seh',states)
        preds = torch.tensor(preds, dtype=self.float)
        preds = self.flat_reward_transform(preds).clip(1e-4, 100).reshape((-1, 1))
        return FlatRewards(preds).view(-1)