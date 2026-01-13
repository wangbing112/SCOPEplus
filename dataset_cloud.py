import json
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from matbench.bench import MatbenchBenchmark
from transformers import BertTokenizer
from sklearn.preprocessing import StandardScaler
import functools
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

class CloudDataset(Dataset):
    def __init__(self, df, tokenizer, blocksize, mu=0.0, std=1.0, train=True):
        """
        df: input datafram
        tokenizer: used for tokenization
        blocksize: the max number of tokens
        mu: mean value of targets
        std: standard deviation of targets
        train: train or validation mode
        """
        self.df = df
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.is_train = train

        if self.is_train:
            self.mu = np.mean(self.df.values[:, 1])
            self.std = np.std(self.df.values[:, 1])
        else:
            self.mu = mu
            self.std = std
        self.df.iloc[:, 1] = (self.df.iloc[:, 1] - self.mu) / self.std
    
        # self.df = self.df.iloc[:5, :]
        print("Number of data:", self.df.shape[0])

    def __len__(self):
        self.len = len(self.df)
        return self.len

    # Cache data for faster training
    @functools.lru_cache(maxsize=None)  # noqa: B019
    def __getitem__(self, index):
        gen_str = self.df.iloc[index, 0]
        target = self.df.iloc[index, 1]

        encoding = self.tokenizer(
            gen_str,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            target=target
        )
    
class CloudDatasetWrapper(object):
    def __init__(
        self, dataset_name, batch_size, vocab_path, blocksize, map_path, num_workers, seed, valid_size, use_phys_params=False
    ):
        super(object, self).__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        # Use extended vocab if use_phys_params is True
        vocab_file = vocab_path if not use_phys_params else vocab_path.replace('tokenizer_cloud_optimade', 'tokenizer_cloud_optimade/vocab_phys.txt') if 'vocab.txt' not in vocab_path else vocab_path.replace('vocab.txt', 'vocab_phys.txt')
        if use_phys_params and not os.path.exists(vocab_file):
            vocab_file = vocab_path  # Fallback to original
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path if not use_phys_params else vocab_path, do_lower_case=False)
        # Manually add new tokens if using phys params
        if use_phys_params:
            try:
                # Read original vocab to get existing tokens
                with open(vocab_path + '/vocab.txt', 'r') as f:
                    existing_tokens = set([line.strip() for line in f if line.strip()])
                # Read extended vocab
                with open('tokenizer_cloud_optimade/vocab_phys.txt', 'r') as f:
                    all_tokens = [line.strip() for line in f if line.strip()]
                # Add only new tokens
                new_tokens = [t for t in all_tokens if t not in existing_tokens]
                if new_tokens:
                    self.tokenizer.add_tokens(new_tokens)
                    print(f"Added {len(new_tokens)} new physical parameter tokens to tokenizer")
            except Exception as e:
                print(f"Warning: Could not add physical parameter tokens: {e}")
        self.blocksize = blocksize
        f = open(map_path, )
        self.map_file = json.load(f)
        self.seed = seed
        self.valid_size = valid_size
        self.use_phys_params = use_phys_params
        # Load generator and wyckoff dict for SCOPE+ generation
        if use_phys_params:
            with open('data/generator.json', 'r') as fp:
                self.gen_str = json.load(fp)
            with open("data/wyckoff-position-multiplicities.json") as file:
                self.wyckoff_multiplicity_dict = json.load(file)

    def extract_lattice_tokens(self, struct):
        """Extract and discretize lattice parameters"""
        lattice = struct.lattice
        # print("lattice:", lattice)
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        
        tokens = []
        tokens.append(f"LA_{int(np.round(np.clip(a, 2.0, 20.0) * 10))}")
        tokens.append(f"LB_{int(np.round(np.clip(b, 2.0, 20.0) * 10))}")
        tokens.append(f"LC_{int(np.round(np.clip(c, 2.0, 20.0) * 10))}")
        tokens.append(f"AA_{int(np.round(np.clip(alpha, 60.0, 180.0)))}")
        tokens.append(f"AB_{int(np.round(np.clip(beta, 60.0, 180.0)))}")
        tokens.append(f"AG_{int(np.round(np.clip(gamma, 60.0, 180.0)))}")
        
        return tokens

    def extract_wyckoff_free_params(self, struct, symm_dataset):
        """Extract Wyckoff free parameters (fractional coordinates)"""
        free_params = []
        wyckoff_positions = symm_dataset['wyckoffs']
        
        # Get unique Wyckoff positions
        unique_wyckoffs = {}
        for i, wyckoff in enumerate(wyckoff_positions):
            if wyckoff not in unique_wyckoffs:
                unique_wyckoffs[wyckoff] = i
        
        # Extract fractional coordinates for unique Wyckoff positions
        for wyckoff, idx in unique_wyckoffs.items():
            site = struct[idx]
            frac_coords = site.frac_coords
            # print("frac_coords:", frac_coords)
            # Discretize each coordinate
            for coord in frac_coords:
                bucket = int(np.round(np.clip(coord, 0.0, 1.0) * 100))
                free_params.append(f"WX_{bucket}")
        
        return free_params

    def generate_scope_plus(self, struct):
        """Generate SCOPE+ string with physical parameters"""
        # print("struct:",struct)
        analyzer = SpacegroupAnalyzer(struct)
        # print("analyzer:",analyzer)
        symm_dataset = analyzer.get_symmetry_dataset()
        wyckoff_positions = symm_dataset['wyckoffs']
        spg_num = str(analyzer.get_space_group_number())
        seq = " ".join(self.gen_str[spg_num])
        
        # Extract lattice parameters
        lattice_tokens = self.extract_lattice_tokens(struct)
        
        # Extract Wyckoff free parameters
        wyckoff_free_params = self.extract_wyckoff_free_params(struct, symm_dataset)
        
        wyckoff_ls = []
        for i in range(len(wyckoff_positions)):
            multiplicity = self.wyckoff_multiplicity_dict[spg_num][wyckoff_positions[i]]
            wyckoff_symbol = multiplicity + wyckoff_positions[i]
            if wyckoff_symbol not in wyckoff_ls:
                wyckoff_ls.append(wyckoff_symbol)
        
        # New SCOPE+ format
        seq = seq + ' | LATTICE: ' + ' '.join(lattice_tokens)
        seq = seq + ' | WYCKOFF: ' + ' '.join(wyckoff_ls) + ' ' + ' '.join(wyckoff_free_params)
        
        comp_ls = []
        for element, ratio in struct.composition.fractional_composition.get_el_amt_dict().items():
            ratio = str(np.round(ratio, 2))
            comp_ls.append(element)
            comp_ls.append(ratio)
        
        seq = seq + ' | ' + ' '.join(comp_ls)
        return seq

    def load_dataset(self, dataset_name, fold, is_train):
        if "matbench" in dataset_name:
            mb = MatbenchBenchmark(autoload=False,subset=[dataset_name])
            for task in mb.tasks:
                task.load()
                if is_train:   
                    df = task.get_train_and_val_data(fold, as_type="df")
                else:
                    df = task.get_test_data(fold, as_type="df", include_target=True)
            
            if self.use_phys_params:
                # Generate SCOPE+ dynamically from structures
                gen_str_list = []
                for idx in df.index:
                    try:
                        # Get structure from dataframe
                        struct = df.loc[idx, 'structure']
                        gen_str_list.append(self.generate_scope_plus(struct))
                    except Exception as e:
                        # Fallback to original gen_str
                        gen_str_list.append(self.map_file.get(idx, ""))
                    # print("gen_str_list:",gen_str_list)
                df["gen_str"] = gen_str_list
            else:
                gen_str = list(map(self.map_file.get, list(df.index)))
                df["gen_str"] = gen_str
            columns_titles = [df.columns[-1], df.columns[1]]
            df = df.reindex(columns=columns_titles)
        elif "unconvbench" in dataset_name:
            if is_train:   
                df = pd.read_csv(os.path.join("data/unconv", dataset_name, str(fold), "train_and_val.csv"))
            else:
                df = pd.read_csv(os.path.join("data/unconv", dataset_name, str(fold), "test.csv"))
        else:
            if is_train:
                dataset_name = dataset_name + "_{}_train.csv".format(fold)
            else:
                dataset_name = dataset_name + "_{}_valid.csv".format(fold)
            df = pd.read_csv(dataset_name)
        
        return df

    def get_data_loaders(self, fold):
        self.df_train = self.load_dataset(self.dataset_name, fold, is_train=True)
        self.df_test = self.load_dataset(self.dataset_name, fold, is_train=False)
        num_data = self.df_train.shape[0]
        indices = list(self.df_train.index)
        random_state = np.random.RandomState(seed=self.seed)
        random_state.shuffle(indices)
        split = int(np.floor(self.valid_size * num_data))
        val_idx = indices[:split]
        self.df_valid = self.df_train.loc[val_idx, :].reset_index(drop=True)

        train_dataset = CloudDataset(self.df_train, self.tokenizer, self.blocksize)
        valid_dataset = CloudDataset(self.df_valid, self.tokenizer, self.blocksize, 
                                  train_dataset.mu, train_dataset.std, train=False)
        test_dataset = CloudDataset(self.df_test, self.tokenizer, self.blocksize, 
                                  train_dataset.mu, train_dataset.std, train=False)
        train_loader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)
        return train_dataset, valid_dataset, test_dataset, train_loader, valid_loader, test_loader