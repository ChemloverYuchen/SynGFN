import uuid
import itertools
import time
from copy import deepcopy
from copy import copy as shallowcopy
from typing import List, Optional, Tuple, Union
from textwrap import dedent
import random
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import json
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.distributions import Categorical, Bernoulli
from torchtyping import TensorType

import os

tmp_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(tmp_path))))

from gflownet.gflownet.utils.common import copy, set_device, set_float_precision, tbool, tfloat
from gflownet.data.template import smarts_list, name_list

global smiles_list, mask_dict, vocab_2, inverse_lookup_2, lookup_2, react_key

with gzip.GzipFile('data/final/smiles_list_mw150_1k.json.gz','r') as f_in:
    smiles_list = json.load(f_in)
with gzip.GzipFile('data/final/mask_dict_mw150_1k.json.gz','r') as f_in:
    mask_dict=json.load(f_in) 
    
vocab_2 = smiles_list + ["EOS","UNI","PAD"]
inverse_lookup_2 = {i: a for (i, a) in enumerate(vocab_2)}
lookup_2 = {a: i for (i, a) in enumerate(vocab_2)}
react_key = list(mask_dict.keys())

CMAP = mpl.colormaps["cividis"]

class BuildingBlock:
    def __init__(
        self,
        device: str = "cpu",#cuda
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        reward_min: float = 1e-8,
        reward_beta: float = 1.0,
        reward_norm: float = 1.0,
        reward_norm_std_mult: float = 0.0,
        reward_func: str = "identity",
        energies_stats: List[int] = None,
        denorm_proxy: bool = False,
        proxy=None,
        oracle=None,
        proxy_state_format: str = "oracle",
        skip_mask_check_1: bool = False,
        skip_mask_check_2: bool = False,
        fixed_distribution_1: Optional[dict] = None,
        fixed_distribution_2: Optional[dict] = None,
        random_distribution_1: Optional[dict] = None,
        random_distribution_2: Optional[dict] = None,
        max_seq_length=7,
        min_seq_length=1,
        **kwargs,
    ):
        self.mol_dict = {}
        self.env_id = env_id
        # Device
        self.device = set_device(device)
        
        # Float precision
        self.float = set_float_precision(float_precision)
        
        #state length
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        
        # Reward settings
        self.min_reward = reward_min
        assert self.min_reward > 0
        self.reward_beta = reward_beta
        assert self.reward_beta > 0
        self.reward_norm = reward_norm
        assert self.reward_norm > 0
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        
        # Proxy and oracle
        self.proxy = proxy
        self.setup_proxy()
        if oracle is None:
            self.oracle = self.proxy
        else:
            self.oracle = oracle
            
        self.proxy_factor = 1.0
        self.proxy_state_format = proxy_state_format
        
        # Flag to skip checking if action is valid (computing mask) before step
        self.skip_mask_check_1 = skip_mask_check_1
        self.skip_mask_check_2 = skip_mask_check_2
        
        self.fixed_distribution_1 = fixed_distribution_1
        self.fixed_distribution_2 = fixed_distribution_2
        self.random_distribution_1 = random_distribution_1
        self.random_distribution_2 = random_distribution_2
        
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        
        # Max trajectory length
        #self.max_traj_length = self.get_max_traj_length()
        
        # Mask dict for action_2
        # self.mask_dict = mask_dict
        
        #state set
        self.smarts_list = smarts_list
        self.name_list = name_list
        self.uni_list = [8, 49, 50, 63, 64, 79, 80, 81, 82, 83, 84, 85, 86] #uni_reaction_idx
        
        self.vocab_1 = smarts_list + ["EOS"]
        self.lookup_1={a: i for (i, a) in enumerate(self.vocab_1)}
        self.inverse_lookup_1 = {i: a for (i, a) in enumerate(self.vocab_1)}
        self.n_alphabet_1 = len(smarts_list)
        self.n_alphabet_2 = len(smiles_list)
        
        self.padding_idx = lookup_2["PAD"]
        self.uni_idx = lookup_2["UNI"]
        self.eos_1 = self.lookup_1["EOS"]
        self.eos_2 = lookup_2["EOS"]
        
        self.source = [self.padding_idx] * self.max_seq_length
        
        self.is_state_list = True
        if isinstance(self.source, TensorType):
            self.is_state_list = False
        # Action space
        self.action_space_1 = self.get_action_space_1()
        self.action_space_2 = self.get_action_space_2()
        
        self.action_space_dim_1 = len(self.action_space_1)
        self.action_space_dim_2 = len(self.action_space_2)
        
        # Call reset() to set initial state, done, n_actions
        self._reset()
        
        # Policy outputs
        self.fixed_policy_output_1 = self.get_policy_output_1(self.fixed_distribution_1)
        self.fixed_policy_output_2 = self.get_policy_output_2(self.fixed_distribution_2)
        self.random_policy_output_1 = self.get_policy_output_1(self.random_distribution_1)
        self.random_policy_output_2 = self.get_policy_output_2(self.random_distribution_2)
        
        self.policy_output_dim_1 = len(self.fixed_policy_output_1)
        self.policy_output_dim_2 = len(self.fixed_policy_output_2)
        
        self.policy_input_dim_1 = len(self._state2policy_1())
        self.policy_input_dim_2 = len(self.state2policy_2())

    def get_action_space_1(self):
        """
        Constructs list with all possible actions
        """
        alphabet = [a for a in range(self.n_alphabet_1)]
        actions = [(el[0], 0) for el in itertools.product(alphabet, repeat=1)]
        actions = actions + [(len(actions), 0)]
        return actions
    
    def get_action_space_2(self):
        alphabet = [a for a in range(self.n_alphabet_2)]
        actions = [el for el in itertools.product(alphabet, repeat=1)]
        actions = actions + [(len(actions),)]+[(len(actions)+1,)]#add EOS and UNI
        return actions
    
    def get_state(self):
        return self.state
    
    def get_padding_idx(self):
        return self.padding_idx
    
    def set_done(self):
        self.done = True
        return self 
      
    def _get_state(self, state: Union[List, TensorType["state_dims"]]):
        if state is None:
            state = copy(self.state)
        return state
    
    def _get_done(self, done: bool):
        if done is None:
            done = self.done
        return done
    
    def get_patt(self, smart):
        try:
            mol = self.mol_dict[smart]
            return mol
        except:
            mol = Chem.MolFromSmarts(smart)
            self.mol_dict[smart] = mol
            return mol
    
    def check_mask_1(self,smiles,smarts):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            return False
        # if bi-reaction
        if '.' in smarts:
            reactants, _ = smarts.split('>>')
            r1_smarts, r2_smarts = reactants.split('.')
            if mol.HasSubstructMatch(self.get_patt(r1_smarts)) or mol.HasSubstructMatch(self.get_patt(r2_smarts)):
                return True
        # if uni-reaction
        else:
            r1_smarts, _ = smarts.split('>>')
            if mol.HasSubstructMatch(self.get_patt(r1_smarts)):
                return True
        return False
    
    def get_mask_invalid_actions_forward_1(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        if state is None:
            state = self.state.copy()  
        if done is None:
            done = self.done
        if done or state[2] != self.padding_idx:
            mask = [True] * (len(self.action_space_1) - 1)+[False]
            return mask
        # if initial state
        if state[0] == self.padding_idx:
            mask=[False] * (len(self.action_space_1) - 1)+[True]
            return mask
        mask = [True] * len(self.action_space_1)
        smi = self._state2readable(state)
        for idx, _ in enumerate(self.action_space_1[:-1]):
            if self.check_mask_1(smi,self.smarts_list[idx]):
                mask[idx] = False
        mask[-1]=False
        return mask
    
    def _get_mask_invalid_actions_forward_2(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action_1: Optional[Tuple] = None,
    ) -> List:
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if isinstance(state,tuple):
            raise ValueError(f"Expected state to be a list, but got a tuple: {state}")
        if done or state[2] != self.padding_idx:
            mask = [True] * len(self.action_space_2)
            mask[-2] = False
            return mask
        if state[0] == self.padding_idx:     
            mask = [False] * len(self.action_space_2)
            mask[-2:] = [True,True]
            return mask
        if action_1[0] == self.eos_1:
            mask = [True] * len(self.action_space_2)
            mask[-2] = False
            return mask
        if action_1[0] in self.uni_list:
            mask = [True] * len(self.action_space_2)
            mask[-2:] = [False, False]
            return mask
        # if bi-reaction
        else:
            smarts = self.inverse_lookup_1[action_1[0]]
            smiles = self._state2readable(state)
            mask = [True] * len(self.action_space_2)
            try:
                mol = Chem.MolFromSmiles(smiles)
            except:
                raise ValueError(
                    f" {smiles} is not a valid smiles."
                )
            reactants, _ = smarts.split('>>')
            r1_smarts, r2_smarts = reactants.split('.')   
            if mol.HasSubstructMatch(self.get_patt(r1_smarts)):
                mask_key = self.name_list[action_1[0]]+"_reactant_2"
                valid_list = mask_dict[mask_key]
                for i in valid_list:
                    mask[i]=False
                mask[-2] = False
                return mask
            elif mol.HasSubstructMatch(self.get_patt(r2_smarts)):
                mask_key = self.name_list[action_1[0]]+"_reactant_1"
                valid_list = mask_dict[mask_key]
                for i in valid_list:
                    mask[i]=False
                mask[-2] = False
                return mask
            else:
                raise ValueError(
                    f" {smarts} is not a valid smarts for {smiles}."
                )
        
    def get_mask_invalid_actions_backward_1(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a_1: Optional[List] = None,
    ) -> List:
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.action_space_dim_1
        if parents_a_1 is None:
            _, parents_a_1, _ = self.get_parents(state, done)
        if len(parents_a_1) == 0:
            mask = [False] *self.action_space_dim_1
            mask[-1] = True
            return mask
        idx = parents_a_1[0]
        if idx[0] == self.padding_idx:
            print('The current state is uninitialized, keeping it invalid')
            print(f"The state that caused the error is {state}")
        else:
            try:
                mask[idx[0]] = False
            except:
                print(f'Assignment error, {idx} is not in the action1 space, mask index out of bounds, mask length: {len(mask)}')
        return mask
    
    def get_mask_invalid_actions_backward_2(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a_2: Optional[List] = None,
    ) -> List:
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True ]*self.action_space_dim_2
        if parents_a_2 is None:
            _, _, parents_a_2 = self.get_parents(state, done)
        mask[parents_a_2[0][0]] = False
        return mask
    
    def statebatch2oracle(self, states: List[List]):
        product_smiles_list = [self._state2readable(s) for s in states]
        return product_smiles_list
    
    def statetorch2oracle(self, states: List[TensorType["batch", "state_dim"]]):
        states = states.tolist()
        return self.statebatch2oracle(states)
    
    def statetorch2readable(self, states):
        states = states.tolist()
        return [self._state2readable(s) for s in states]
    

    def _state2policy_1(self, state: List = None):
        if state is None:
            state = self.state.copy()
        state_readable = self._state2readable(state)
        if state[0] == self.padding_idx or state_readable is None:
            state_policy = np.zeros(4096, dtype=np.float32) + 0.1 * np.random.normal(0, 1, (4096,))
        else:
            mol = Chem.MolFromSmiles(state_readable)
            features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol,3,4096)
            state_policy = np.array(features_vec).astype(np.float32)
        return state_policy   
    
    def label2onehot(self,label:str):
        label_list = react_key + ["EOS"] + ["UNI"]
        label_to_index = {tag: i for (i,tag) in enumerate(label_list)}
        label_onehot = np.zeros(len(label_list), dtype=np.float32)
        index = label_to_index[label]
        label_onehot[index] = 1.0
        return label_onehot   
    
    def state2policy_2(self,state: List = None, action_1: Tuple =None):
        if state is None:
            state = self.state.copy()
        label_list = react_key + ["EOS"] + ["UNI"]
        state_readable = self._state2readable(state)
        if state[0] == self.padding_idx or state_readable is None:
            label_onehot = np.zeros(len(label_list), dtype=np.float32) + 0.1 * np.random.normal(0, 1, (len(label_list),))
            features_vec = np.zeros(4096, dtype=np.float32) + 0.1 * np.random.normal(0, 1, (4096,))
            state_policy = np.hstack((features_vec, label_onehot)).astype(np.float32)
            return state_policy
        mol = Chem.MolFromSmiles(state_readable)
        label_onehot = np.zeros(len(label_list), dtype=np.float32) + 0.1 * np.random.normal(0, 1, (len(label_list),))
        if action_1[0] in self.uni_list:
            label="UNI"
            label_onehot = self.label2onehot(label)
        elif action_1[0] == self.eos_1 or self.done:
            label="EOS"
            label_onehot = self.label2onehot(label)
        elif action_1[0] == self.eos_1 +1:
            label="UNI"
            label_onehot = self.label2onehot(label)
        else:
            smarts = self.inverse_lookup_1[action_1[0]]
            reactants, _ = smarts.split('>>')
            r1_smarts, r2_smarts = reactants.split('.')
            if mol.HasSubstructMatch(self.get_patt(r1_smarts)):
                label = self.name_list[action_1[0]]+"_reactant_2"
                label_onehot = self.label2onehot(label)
            elif mol.HasSubstructMatch(self.get_patt(r2_smarts)):
                label = self.name_list[action_1[0]]+"_reactant_1"
                label_onehot = self.label2onehot(label)        
        features_vec = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 3, 4096)
        features_vec = np.array(features_vec, dtype=np.float32)
        state_policy = np.hstack((features_vec, label_onehot)).astype(np.float32)
            
        return state_policy
    
    def statebatch2policy_1(self, states: List[List]):
        return [self._state2policy_1(s) for s in states]

    def _statetorch2policy_1(self, states: List[TensorType["batch", "state_dim"]]):
        if not isinstance(states, list):
            states = states.tolist()
        return self.statebatch2policy_1(states)
    
    def statebatch2policy_2(self,states: List[List], actions: List[Tuple[int]]):
        return [self.state2policy_2(s, a) for s, a in zip(states, actions)] 
    
    def get_uni_product(self,smiles1,smarts,index):
        mol = Chem.MolFromSmiles(smiles1)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol,))
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        if len(uniqps) == 0:
            return None
        uniqps_sort = sorted(uniqps.values())        
        smiles = uniqps_sort[int(index)]
        return smiles
    
    def get_bi_product(self,smiles_1,smiles_2,smarts,index):
        mol_1 = Chem.MolFromSmiles(smiles_1)
        mol_2 = Chem.MolFromSmiles(smiles_2)
        rxn = AllChem.ReactionFromSmarts(smarts)
        ps = rxn.RunReactants((mol_1,mol_2))+rxn.RunReactants((mol_2,mol_1))
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        if len(uniqps) == 0:
            return None
        uniqps_sort = sorted(uniqps.values())
        smiles = uniqps_sort[int(index)]
        return smiles

    def _state2readable(self, state: List):
        if state[0] == self.padding_idx:
            return None
        #if only one block
        elif state[1] == self.padding_idx:
            smiles = inverse_lookup_2[state[0]]
            return smiles
        #two blocks,return intermediate product
        elif state[2] == self.padding_idx:
            #if uni-reaction
            if state[3] in self.uni_list and state[4] != -1:
                return self.get_uni_product(inverse_lookup_2[state[0]],self.inverse_lookup_1[state[3]],state[4])
            elif state[3] in self.uni_list and state[4] == -1:
                return inverse_lookup_2[state[0]]
            #if bi-reaction
            elif state[3] not in self.uni_list and state[4] != -1:
                return self.get_bi_product(inverse_lookup_2[state[0]],inverse_lookup_2[state[1]],self.inverse_lookup_1[state[3]],state[4])
            else:
                return inverse_lookup_2[state[0]]
        #three blocks,return final product
        elif state[2] != self.padding_idx:
            #check the intermediate product is uni or bi
            if state[3] in self.uni_list and state[4] != -1:
                intermediate_smiles = self.get_uni_product(inverse_lookup_2[state[0]],self.inverse_lookup_1[state[3]],state[4])
            elif state[3] not in self.uni_list and state[4] != -1:
                intermediate_smiles = self.get_bi_product(inverse_lookup_2[state[0]],inverse_lookup_2[state[1]],self.inverse_lookup_1[state[3]],state[4])
            else:
                intermediate_smiles = inverse_lookup_2[state[0]]
                
            #check the final product is uni or bi
            if state[5] in self.uni_list and state[6] != -1:
                final_smiles = self.get_uni_product(intermediate_smiles,self.inverse_lookup_1[state[5]],state[6])                
                return final_smiles
            # elif state[5] in self.uni_list and state[6] == -1:
            #     return intermediate_smiles
            elif state[5] not in self.uni_list and state[6] != -1:
                final_smiles = self.get_bi_product(intermediate_smiles,inverse_lookup_2[state[2]],self.inverse_lookup_1[state[5]],state[6])
                return final_smiles
            else:
                return intermediate_smiles
        else:
            return None


        
        
    def statebatch2proxy(self, states: List[List]):
        return [self._state2readable(s) for s in states]
    
    def get_parents(self, state=None, done=None, action=None):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        parents = []
        actions_1 = []
        actions_2 = []
        #if one block
        if state[0] != self.padding_idx and state[1] == self.padding_idx:
            parent_action_2 = tuple(state[0:1])
            actions_2.append(parent_action_2)
            return [self.source], [(0,)], actions_2
        
        #if two blocks
        elif state[1] != self.padding_idx and state[2] == self.padding_idx:
            parent_actions_1 = tuple(state[3:4])
            actions_1.append(parent_actions_1)
            parent_action_2 = tuple(state[1:2])
            actions_2.append(parent_action_2)
            parent = state.copy()
            parent[1] = self.padding_idx
            parent[3] = self.padding_idx
            parent[4] = self.padding_idx
            parents.append(parent)
            return parents,actions_1,actions_2
        
        #if three blocks
        else:
            parent_actions_1 = tuple(state[5:6])
            actions_1.append(parent_actions_1)

            parent_action_2 = tuple(state[2:3])
            actions_2.append(parent_action_2)

            parent = state.copy()
            parent[2] = self.padding_idx
            parent[5] = self.padding_idx
            parent[6] = self.padding_idx
            parents.append(parent)
            return parents, actions_1, actions_2
            
    def _pre_step(
        self, action_1: Tuple[int], action_2: Tuple[int], backward: bool = False,
    ) -> Tuple[bool, List[int], Tuple[int]]:
        if action_1 not in self.action_space_1:
            raise ValueError(
                f"Tried to execute action {action_1} not present in action space."
            )
        if action_2 not in self.action_space_2:
            raise ValueError(
                f"Tried to execute action {action_2} not present in action space."
            )
        if backward is True:
            if self.done:
                return False
            elif self.get_mask_invalid_actions_backward_1()[action_1[0]] or self.get_mask_invalid_actions_backward_2()[action_2[0]]:
                return False
            else:
                return True
        # If forward and env is done, step should not proceed.
        else:
            if self.done or self.state[2] != self.padding_idx:
                return False
            elif self.state[0] == self.padding_idx:
                if self._get_mask_invalid_actions_forward_2()[action_2[0]]:
                    return False
                else:
                    return True
            else:
                if self.get_mask_invalid_actions_forward_1()[action_1[0]] or self._get_mask_invalid_actions_forward_2(action_1=action_1)[action_2[0]]:
                    return False
                else:
                    return True

    def get_uni_product_idx(self,smiles,action_1):
        if action_1 == self.eos_1:
            return -1
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return -1
        rxn = AllChem.ReactionFromSmarts(self.inverse_lookup_1[action_1])
        ps = rxn.RunReactants((mol,))
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        uniqps_sort = sorted(uniqps.values())
        if len(uniqps_sort) == 0:
            return -1
        return random.randrange(len(uniqps_sort))
    
    def get_bi_product_idx(self,smiles,action_1,action_2):
        if action_1 == self.eos_1 or action_2 == self.eos_2:
            return -1
        mol_1 = Chem.MolFromSmiles(smiles)
        if mol_1 is None:
            return -1
        mol_2 = Chem.MolFromSmiles(inverse_lookup_2[action_2])
        assert mol_2 is not None
        rxn = AllChem.ReactionFromSmarts(self.inverse_lookup_1[action_1])
        ps = rxn.RunReactants((mol_1,mol_2))+ rxn.RunReactants((mol_2,mol_1))
        if len(ps) == 0:
            return -1
        uniqps = {}
        for p in ps:
            try:
                Chem.SanitizeMol(p[0])
                inchi = Chem.MolToInchi(p[0])
                uniqps[inchi] = Chem.MolToSmiles(p[0])
            except:
                pass
        uniqps_sort=sorted(uniqps.values())
        if len(uniqps_sort) == 0:
            return -1
        return random.randrange(len(uniqps_sort))
    

    def _step(self, action_1: Tuple[int], action_2: Tuple[int],backward: bool = False):
        assert action_1 in self.action_space_1
        assert action_2 in self.action_space_2
        if backward is True:
            if action_1[0] == self.eos_1 and action_2[0] == self.eos_2:
                self.n_actions += 1
                self.done = False
                return self.state, action_1, action_2, True
            parents, _, _ = self.get_parents()
            
            done = True if parents[0] == self.source else False
            state_next = parents[0]
            self.set_state(state_next, done=done)
            self.n_actions += 1
            return self.state, action_1, action_2, True

        # if forward
        else:
            do_step = self._pre_step(action_1, action_2, backward=False)
            self.done = True 
            if not do_step:
                self.set_state(self.state,done=True)
                self.n_actions += 1
                return self.state,action_1,action_2, True
            # if initial state and action_2 is not eos    
            elif self.state[0] == self.padding_idx and action_2[0] != self.eos_2:
                assert action_2[0] !=self.uni_idx
                state_next = self.state.copy()
                state_next[0] = action_2[0]
                self.set_state(state_next, done=False)
                self.n_actions += 1
                return self.state, action_1, action_2, True
            elif self.state[0] != self.padding_idx and (action_1[0] == self.eos_1 or action_2[0] == self.eos_2):
                self.set_state(self.state,done=True)
                self.n_actions += 1
                return self.state, action_1, action_2, True
            #condition1:only one block,need to add second block to form intermidiate product
            elif self.state[1] == self.padding_idx and action_1[0] in self.uni_list:
                state_next = self.state.copy()
                state_next[4] = self.get_uni_product_idx(self._state2readable(state_next),action_1[0])
                if state_next[4] == -1:
                    self.set_state(self.state, done=True)
                    valid = False 
                    return self.state, action_1, action_2, valid                
                else:
                    state_next[1] = action_2[0]
                    state_next[3] = action_1[0]
                    valid = True
                    self.set_state(state_next, done=False)
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
            elif self.state[1] == self.padding_idx and action_1[0] not in self.uni_list:
                state_next = self.state.copy()
                state_next[4] = self.get_bi_product_idx(self._state2readable(state_next),action_1[0],action_2[0])
                if state_next[4] == -1:
                    self.set_state(self.state,done=True)
                    valid = False
                    return self.state, action_1, action_2, valid
                else:
                    state_next[1] = action_2[0]
                    state_next[3] = action_1[0]    
                    valid = True
                    self.set_state(state_next, done=False)
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
            #condition2:two block,need to add third block to form final product
            elif self.state[2] == self.padding_idx and action_1[0] in self.uni_list:
                state_next = self.state.copy()
                state_next[6] = self.get_uni_product_idx(self._state2readable(state_next),action_1[0])
                if state_next[6] == -1:
                    self.set_state(self.state,done=True)
                    valid = False
                    return self.state, action_1, action_2, valid
                else:
                    state_next[2] = action_2[0]
                    state_next[5] = action_1[0]
                    valid = True
                    self.set_state(state_next, done=False)
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
            elif self.state[2] == self.padding_idx and action_1[0] not in self.uni_list:
                state_next = self.state.copy()
                state_next[6] = self.get_bi_product_idx(self._state2readable(state_next),action_1[0],action_2[0])
                if state_next[6] == -1:
                    self.set_state(self.state,done=True)
                    valid = False
                    return self.state, action_1, action_2, valid
                else:
                    state_next[2] = action_2[0]
                    state_next[5] = action_1[0]
                    valid = True
                    self.set_state(state_next, done=False)
                    self.n_actions += 1
                    return self.state, action_1, action_2, valid
            else:
                    raise ValueError(f"step() should not be called and state is {self.state}") 
                                             
    def sample_actions_batch_1(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_1"],
        mask: Optional[TensorType["n_states", "policy_output_dim_1"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 3,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        if is_backward:
            actions = [self.action_space_1[list(m).index(False)] for m in mask]
            return actions, None
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "random":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        assert not torch.all(mask), dedent(
            """
        All actions in the mask are invalid.
        """
        )
        logits[mask] = -torch.inf
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        # Build actions
        actions = [self.action_space_1[idx] for idx in action_indices]
        return actions, None
    
    def sample_actions_batch_2(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_2"],
        mask: Optional[TensorType["n_states", "policy_output_dim_2"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        #sample_only: Optional[bool] = False,
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        device = policy_outputs.device
        if is_backward:
            dt = np.arange(mask.shape[1])
            actions = [(dt[~m][0],) for m in mask]
            return actions, None
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "random":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        assert not torch.any(torch.all(mask,axis=1)), dedent(
                """
            All actions in the mask are invalid.
            """
            )
        logits[mask] = -torch.inf
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            if torch.isnan(logits).any():
                raise ValueError(f"{mask}")
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        # Build actions
        actions = list(zip(action_indices.tolist()))
        return actions, None
    
    def get_logprobs_1(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_1"],
        is_forward: bool,
        actions: TensorType["n_states", "actions_dim_1"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask: TensorType["batch_size", "policy_output_dim_1"] = None,
    ) -> TensorType["batch_size"]:
        device = policy_outputs.device
        if is_forward:
            ns_range = torch.arange(policy_outputs.shape[0]).to(device)
            logits = policy_outputs
            if mask is not None:
                logits[mask] = -torch.inf
            action_indices = (
                torch.tensor(
                    actions[:,0]
                )
                .to(int)
                .to(device)
            )
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        else:
            logprobs = torch.zeros(policy_outputs.shape[0], dtype=self.float, device=device)
        return logprobs
    
    def get_logprobs_2(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim_2"],
        is_forward: bool,
        actions: TensorType["n_states", "actions_dim_2"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask: TensorType["batch_size", "policy_output_dim_2"] = None,
    ) -> TensorType["batch_size"]:
        device = policy_outputs.device
        if is_forward:
            ns_range = torch.arange(policy_outputs.shape[0]).to(device)
            logits = policy_outputs
            if mask is not None:
                logits[mask] = -torch.inf
            action_indices = (
                torch.tensor(
                    actions[:,0]
                )
                .to(int)
                .to(device)
            )
            logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        else:
            logprobs = torch.zeros(policy_outputs.shape[0], dtype=self.float, device=device)
        return logprobs
    
    def get_policy_output_1(self, params: Optional[dict] = None):
        return np.ones(self.action_space_dim_1)
    
    def get_policy_output_2(self, params: Optional[dict] = None):
        return np.ones(self.action_space_dim_2)
    
    def policy2state(self, state_policy: List) -> List:
        return state_policy
    
    def readable2state(self, readable):
        return readable
    
    def traj2readable(self, traj=None):
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")
    
    def reward(self, state=None, done=None):
        state = self._get_state(state)
        done = self._get_done(done)
        if done is False:
            return tfloat(0.0, float_type=self.float, device=self.device)
        reward=self.proxy(self.state2proxy(state))
        return self.proxy2reward(reward)                  

    def reward_batch(self, states, done=None):
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)
        if isinstance(states_proxy, torch.Tensor):
            states_proxy = states_proxy[list(done), :]
        elif isinstance(states_proxy, list):
            states_proxy = [states_proxy[i] for i in range(len(done)) if done[i]]
        rewards = np.zeros(len(done))
        if len(states_proxy) > 0:
            proxy_vals = self.proxy(states_proxy)
            rewards[list(done)] = self.proxy2reward(proxy_vals).tolist()
        return rewards
                                 
    
    def proxy2reward(self, proxy_vals):
        if self.denorm_proxy:
            proxy_vals = (
                proxy_vals * (self.energies_stats[1] - self.energies_stats[0])
                + self.energies_stats[0]
            )
        if self.reward_func == "power":
            return 10 * torch.clamp(
                (self.proxy_factor * proxy_vals / self.reward_norm) ** self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "boltzmann":
            return torch.clamp(
                torch.exp(self.proxy_factor * self.reward_beta * proxy_vals),
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "identity":
            return torch.clamp(
                self.proxy_factor * proxy_vals,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "shift":
            return torch.clamp(
                self.proxy_factor * proxy_vals + self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "sigmoid":
            return torch.clamp(
                1 / (1 + torch.exp(0.75 * (proxy_vals + 6.5))),
                min=self.min_reward,
                max=None,
            )
        else:
            raise NotImplementedError
    
    def reward2proxy(self, reward):
        if self.reward_func == "power":
            return self.reward_norm * (reward / (10 * self.proxy_factor)) ** (1/self.reward_beta)
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        elif self.reward_func == "shift":
            return self.proxy_factor * (reward - self.reward_beta)
        elif self.reward_func == "sigmoid":
            return torch.log(1 / reward - 1) / 0.75 -6.5
        else:
            raise NotImplementedError
        
    def _reset(self, env_id: Union[int, str] = None):
        self.state = copy(self.source)
        self.n_actions = 0
        self.done = False
        if env_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = env_id
        return self

    def set_id(self, env_id: Union[int, str]):
        self.id = env_id
        return self

    def set_state(self, state: List, done: Optional[bool] = False):
        self.state = copy(state)
        self.done = done
        return self
    
    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    def make_train_set(
        self,
        ntrain,
        oracle=None,
        seed=168,
        output_csv=None,
    ):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        ntest : int
            Number of test samples.

        seed : int
            Random seed.

        output_csv: str
            Optional path to store the test set as CSV.
        """
        samples_dict = oracle.initializeDataset(
            save=False, returnData=True, customSize=ntrain, custom_seed=seed
        )
        energies = samples_dict["energies"]
        samples_mat = samples_dict["samples"]
        state_letters = oracle.numbers2letters(samples_mat)
        state_ints = [
            "".join([str(el) for el in state if el > 0]) for state in samples_mat
        ]
        if isinstance(energies, dict):
            energies.update({"samples": state_letters, "indices": state_ints})
            df_train = pd.DataFrame(energies)
        else:
            df_train = pd.DataFrame(
                {"samples": state_letters, "indices": state_ints, "energies": energies}
            )
        if output_csv:
            df_train.to_csv(output_csv)
        return df_train

    def make_test_set(
        self,
        path_base_dataset,
        ntest,
        min_length=0,
        max_length=np.inf,
        seed=167,
        output_csv=None,
    ):
        if path_base_dataset is None:
            return None, None
        times = {
            "all": 0.0,
            "indices": 0.0,
        }
        t0_all = time.time()
        if seed:
            np.random.seed(seed)
        df_base = pd.read_csv(path_base_dataset, index_col=0)
        df_base = df_base.loc[
            (df_base["samples"].map(len) >= min_length)
            & (df_base["samples"].map(len) <= max_length)
        ]
        energies_base = df_base["energies"].values
        min_base = energies_base.min()
        max_base = energies_base.max()
        distr_unif = np.random.uniform(low=min_base, high=max_base, size=ntest)
        # Get minimum distance samples without duplicates
        t0_indices = time.time()
        idx_samples = []
        for idx in tqdm(range(ntest)):
            dist = np.abs(energies_base - distr_unif[idx])
            idx_min = np.argmin(dist)
            if idx_min in idx_samples:
                idx_sort = np.argsort(dist)
                for idx_next in idx_sort:
                    if idx_next not in idx_samples:
                        idx_samples.append(idx_next)
                        break
            else:
                idx_samples.append(idx_min)
        t1_indices = time.time()
        times["indices"] += t1_indices - t0_indices
        # Make test set
        df_test = df_base.iloc[idx_samples]
        if output_csv:
            df_test.to_csv(output_csv)
        t1_all = time.time()
        times["all"] += t1_all - t0_all
        return df_test, times
    
    def get_trajectories(
        self, traj_list, traj_actions_list_1, traj_actions_list_2, current_traj, current_actions_1,current_actions_2
    ):
        parents, parents_actions_1, parents_actions_2 = self.get_parents(current_traj[-1], False)
        if parents == []:
            traj_list.append(current_traj)
            traj_actions_list_1.append(current_actions_1)
            traj_actions_list_2.append(current_actions_2)
            return traj_list, traj_actions_list_1,traj_actions_list_2
        for idx, (p, a_1, a_2) in enumerate(zip(parents, parents_actions_1, parents_actions_2)):
            traj_list, traj_actions_list_1, traj_actions_list_2 = self.get_trajectories(
                traj_list, traj_actions_list_1,traj_actions_list_2, current_traj + [p], current_actions_1 + [a_1],current_actions_2 + [a_2]
            )
        return traj_list, traj_actions_list_1, traj_actions_list_2

    def setup_proxy(self):
        if self.proxy:
            self.proxy.setup(self)

    @torch.no_grad()
    def compute_train_energy_proxy_and_rewards(self):
        gt_energy, proxy_energy = self.proxy.infer_on_train_set()
        gt_reward = self.proxy2reward(gt_energy)
        proxy_reward = self.proxy2reward(proxy_energy)

        return gt_energy, proxy_energy, gt_reward, proxy_reward

    @torch.no_grad()
    def top_k_metrics_and_plots(
        self,
        states,
        top_k,
        name,
        energy=None,
        reward=None,
        step=None,
        **kwargs,
    ):
        if states is None and energy is None and reward is None:
            assert name == "train"
            (
                energy,
                proxy,
                energy_reward,
                proxy_reward,
            ) = self.compute_train_energy_proxy_and_rewards()
            name = "train ground truth"
            reward = energy_reward
        elif energy is None and reward is None:
            # TODO: fix this
            x = torch.stack([self.state2proxy(s) for s in states])
            energy = self.proxy(x.to(self.device)).cpu()
            reward = self.proxy2reward(energy)

        assert energy is not None and reward is not None

        # select top k best energies and rewards
        top_k_e = torch.topk(energy, top_k, largest=False, dim=0).values.numpy()
        top_k_r = torch.topk(reward, top_k, largest=True, dim=0).values.numpy()

        # find best energy and reward
        best_e = torch.min(energy).item()
        best_r = torch.max(reward).item()

        # to numpy to plot
        energy = energy.numpy()
        reward = reward.numpy()

        # compute stats
        mean_e = np.mean(energy)
        mean_r = np.mean(reward)

        std_e = np.std(energy)
        std_r = np.std(reward)

        mean_top_k_e = np.mean(top_k_e)
        mean_top_k_r = np.mean(top_k_r)

        std_top_k_e = np.std(top_k_e)
        std_top_k_r = np.std(top_k_r)

        # automatic color scale
        # currently: cividis colour map
        colors = ["full", "top_k"]
        normalizer = mpl.colors.Normalize(vmin=0, vmax=len(colors) - 0.5)
        colors = {k: CMAP(normalizer(i)) for i, k in enumerate(colors[::-1])}

        # two sublopts: left is energy, right is reward
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # energy full distribution and stats lines
        ax[0].hist(
            energy,
            bins=100,
            alpha=0.35,
            label=f"All = {len(energy)}",
            color=colors["full"],
            density=True,
        )
        ax[0].axvline(
            mean_e,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_e:.3f}",
        )
        ax[0].axvline(
            mean_e + std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_e:.3f}",
        )
        ax[0].axvline(
            mean_e - std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # energy top k distribution and stats lines
        ax[0].hist(
            top_k_e,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[0].axvline(
            mean_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e + std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e - std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )
        # energy title & legend
        ax[0].set_title(
            f"Energy distribution for {top_k} vs {len(energy)}"
            + f" samples\nBest: {best_e:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[0].legend()

        # reward full distribution and stats lines
        ax[1].hist(
            reward,
            bins=100,
            alpha=0.35,
            label=f"All = {len(reward)}",
            color=colors["full"],
            density=True,
        )
        ax[1].axvline(
            mean_r,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_r:.3f}",
        )
        ax[1].axvline(
            mean_r + std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_r:.3f}",
        )
        ax[1].axvline(
            mean_r - std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # reward top k distribution and stats lines
        ax[1].hist(
            top_k_r,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[1].axvline(
            mean_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r + std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r - std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )

        # reward title & legend
        ax[1].set_title(
            f"Reward distribution for {top_k} vs {len(reward)}"
            + f" samples\nBest: {best_r:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[1].legend()

        # Finalize figure
        title = f"{name.capitalize()} energy and reward distributions"
        if step is not None:
            title += f" (step {step})"
        fig.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # store metrics
        metrics = {
            f"Mean {name} energy": mean_e,
            f"Std {name} energy": std_e,
            f"Mean {name} reward": mean_r,
            f"Std {name} reward": std_r,
            f"Mean {name} top k energy": mean_top_k_e,
            f"Std {name} top k energy": std_top_k_e,
            f"Mean {name} top k reward": mean_top_k_r,
            f"Std {name} top k reward": std_top_k_r,
            f"Best (min) {name} energy": best_e,
            f"Best (max) {name} reward": best_r,
        }
        figs = [fig]
        fig_names = [title]

        if name.lower() == "train ground truth":
            proxy_metrics, proxy_figs, proxy_fig_names = self.top_k_metrics_and_plots(
                None,
                top_k,
                "train proxy",
                energy=proxy,
                reward=proxy_reward,
                step=None,
                **kwargs,
            )
            metrics.update(proxy_metrics)
            figs += proxy_figs
            fig_names += proxy_fig_names

        return metrics, figs, fig_names

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, oracle=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if title == None:
            title = "Scores of Sampled States"
        if oracle is None:
            oracle = self.oracle
        if scores is None:
            if isinstance(states[0], torch.Tensor):
                states = torch.vstack(states).to(self.device, self.float)
            if isinstance(states, torch.Tensor) == False:
                states = torch.tensor(states, device=self.device, dtype=self.float)
            oracle_states = self.statetorch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def _copy(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in ["action_space_2", "fixed_policy_output_2", "random_policy_output_2"]:
                setattr(result, k, shallowcopy(v))
            else:
                setattr(result, k, deepcopy(v))
        return result