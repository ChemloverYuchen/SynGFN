import argparse
import pandas as pd
from rdkit import Chem
from multiprocessing import Pool
import numpy as np
from rdkit.Chem import Descriptors
import os

def find_func_group(smiles, r1, r2, name):
    func_group = ""
    mol = Chem.MolFromSmiles(smiles)
    if not pd.isna(r1) and mol.HasSubstructMatch(r1):
        func_group += name + "_reactant_1;"
    if not pd.isna(r2) and mol.HasSubstructMatch(r2):
        func_group += name + "_reactant_2;"
    return func_group

def process_row(row):
    smiles, reactants = row
    func_group = ""
    for name, r1, r2 in reactants:
        func_group += find_func_group(smiles, r1, r2, name)
    return func_group

def filter_no_react(df):
    df = df[~df['func_group'].isna() & (df['func_group'] != '')]
    df.reset_index(drop=True, inplace=True)
    return df

def filter_valid_atoms(df, valid_atoms):
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if not all(atom in valid_atoms for atom in atoms):
            df.drop(i, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def filter_by_ring_count(df, max_ring_count):
    index_list = list(df.index)
    for i in index_list:
        smiles = df['SMILES'][i]
        mol = Chem.MolFromSmiles(smiles)
        ring_count = len(Chem.GetSymmSSSR(mol))
        if ring_count > max_ring_count:
            df.drop(i, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def filter_by_max_ring_atoms(df, max_atoms):
    max_ring_atoms = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            max_ring_atoms.append(0)  
        else:
            rings = Chem.GetSymmSSSR(mol)
            ring_atoms = max([len(ring) for ring in rings]) if rings else 0  
            max_ring_atoms.append(ring_atoms)
    df = df[np.array(max_ring_atoms) <= max_atoms]
    df.reset_index(drop=True, inplace=True)
    return df

def filter_by_mw(df, mw):
    if 'Mw' not in df.columns:
        df['Mw'] = df['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
    df = df[df['Mw'].astype(float).astype(int) <= mw-1].copy()
    df.reset_index(drop=True, inplace=True)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a building block library with reaction tags and data processing based on the input building block library and reaction templates.')
    parser.add_argument('--df_bb_path', type=str, default='raw/building_block.csv', help='Path to the building_block CSV file')
    parser.add_argument('--df_r_path', type=str, default='raw/reaction_template.csv', help='Path to the reaction_template CSV file')
    parser.add_argument('--n_processes', type=int, default=1, help='Number of processes to use')
    
    parser.add_argument('--valid_atoms', type=str, nargs='+', default=['C', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'B'], help='List of valid atoms (default: C O N P S F Cl Br I B)')
    parser.add_argument('--max_ring_count', type=int, default=4, help='Maximum number of rings allowed (default: 4)')
    parser.add_argument('--max_ring_atoms', type=int, default=7, help='Maximum number of atoms in the largest ring (default: 7)')
    parser.add_argument('--max_mw', type=float, default=200, help='Maximum molecular weight allowed (default: 200)')
    parser.add_argument('--df_b_r_path', type=str, default='processed/building_block_add_react_tag.csv', help='Path to the building_block added reactant CSV file')

    args = parser.parse_args()
    
    if not os.path.exists(args.df_b_r_path):
        df_bb = pd.read_csv(args.df_bb_path) # The file must provide molecular SMILES
        df_r = pd.read_csv(args.df_r_path) # The file must provide reaction template SMARTS, reaction name/code, and SMARTS for reactant 1 and reactant 2

        df_r['mol_1'] = df_r['reactant1'].apply(lambda x: Chem.MolFromSmarts(x) if not pd.isna(x) else None)
        df_r['mol_2'] = df_r['reactant2'].apply(lambda x: Chem.MolFromSmarts(x) if not pd.isna(x) else None)
        reactants = df_r[['name', 'mol_1', 'mol_2']].values.tolist()

        data = [(row['SMILES'], reactants) for _, row in df_bb.iterrows()]

        with Pool(processes=args.n_processes) as pool:
            results = pool.map(process_row, data)

        df_bb['func_group'] = results
        df_bb.to_csv('processed/building_block_add_react_tag.csv', index=False)
    else:
        df_bb = pd.read_csv(args.df_b_r_path)
        
    # Data post-processing
        df_bb = filter_no_react(df_bb)
        df_bb = filter_valid_atoms(df_bb, valid_atoms=args.valid_atoms)
        df_bb = filter_by_ring_count(df_bb, max_ring_count=args.max_ring_count)
        df_bb = filter_by_max_ring_atoms(df_bb, max_atoms=args.max_ring_atoms)
        df_bb = filter_by_mw(df_bb, mw=args.max_mw)
        output_filename = f"processed/building_block_processed_mw{int(args.max_mw)}.csv"
        df_bb.to_csv(output_filename, index=False)
