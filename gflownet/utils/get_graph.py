from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import numpy as np
import pandas as pd
from torch_geometric.data import Data,Batch

atom_list = ["B", "Br", "C", "Cl", "F", "H", "I", "N", "O", "P", "S", "Si", "Se"]
class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output
    
class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        
    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()
    
    def chiral(self, atom):
        return atom.GetChiralTag().name.lower()
    
    def ring(self, atom):
        if atom.GetIsAromatic():
            return 2
        if atom.IsInRing():
            return 1
        return 0
    
class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()
    
    def stereo(self, bond):
        return bond.GetStereo().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()

atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": set(atom_list),
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
        "chiral": {"chi_unspecified", "chi_tetrahedral_cw", "chi_tetrahedral_ccw"},
        "ring": {0, 1, 2},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "stereo": {"stereonone", "stereoz", "stereoe", "stereocis", "stereotrans"},
        "conjugated": {True, False},
    }
)

def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_smiles(smiles_list):
    data_list = []

    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        data = Data(x=torch.tensor(atom_features, dtype=torch.float),
                    edge_index=torch.tensor(pair_indices, dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(bond_features, dtype=torch.float))
        
        data_list.append(data)

    return Batch.from_data_list(data_list)