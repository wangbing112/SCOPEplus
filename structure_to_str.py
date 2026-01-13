import json
import os
import numpy as np
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core.structure import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import glob
from time import perf_counter

"""
Functions for reduced string representation of a structure
"""

def discretize_lattice_param(value, param_type='length', bins=100):
    """
    Discretize lattice parameters into tokens
    For lengths: range 2-20 Angstrom, 0.1 Angstrom bins -> LA_20 to LA_200
    For angles: range 60-180 degrees, 1 degree bins -> AA_60 to AA_180
    """
    if param_type == 'length':
        # Length in Angstrom: 2-20 A, discretize to 0.1 A precision
        value = np.clip(value, 2.0, 20.0)
        bucket = int(np.round(value * 10))  # 20.0 -> 200, 2.0 -> 20
        return f"LA_{bucket}" if param_type == 'length' and 'a' in param_type else f"LB_{bucket}" if 'b' in param_type else f"LC_{bucket}"
    elif param_type == 'angle':
        # Angle in degrees: 60-180, discretize to 1 degree precision
        value = np.clip(value, 60.0, 180.0)
        bucket = int(np.round(value))
        return f"AA_{bucket}" if 'alpha' in param_type else f"AB_{bucket}" if 'beta' in param_type else f"AG_{bucket}"

def discretize_wyckoff_param(value, bins=100):
    """
    Discretize Wyckoff free parameters (fractional coordinates)
    Range: 0-1, discretize to 0.01 precision -> WX_0 to WX_100
    """
    value = np.clip(value, 0.0, 1.0)
    bucket = int(np.round(value * 100))  # 0.0 -> 0, 1.0 -> 100
    return bucket

def extract_lattice_tokens(struct):
    """Extract and discretize lattice parameters"""
    lattice = struct.lattice
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

def extract_wyckoff_free_params(struct, symm_dataset):
    """Extract Wyckoff free parameters (fractional coordinates)"""
    free_params = []
    wyckoff_positions = symm_dataset['wyckoffs']
    equivalent_atoms = symm_dataset['equivalent_atoms']
    
    # Get unique Wyckoff positions
    unique_wyckoffs = {}
    for i, wyckoff in enumerate(wyckoff_positions):
        if wyckoff not in unique_wyckoffs:
            unique_wyckoffs[wyckoff] = i
    
    # Extract fractional coordinates for unique Wyckoff positions
    for wyckoff, idx in unique_wyckoffs.items():
        site = struct[idx]
        frac_coords = site.frac_coords
        # Discretize each coordinate
        for coord in frac_coords:
            bucket = int(np.round(np.clip(coord, 0.0, 1.0) * 100))
            free_params.append(f"WX_{bucket}")
    
    return free_params

def generate_seq(struct, gen_str, wyckoff_multiplicity_dict):
    analyzer = SpacegroupAnalyzer(struct)
    symm_dataset = analyzer.get_symmetry_dataset()
    wyckoff_positions = symm_dataset['wyckoffs']

    spg_num = str(analyzer.get_space_group_number())
    seq = " ".join(gen_str[spg_num])

    # Extract lattice parameters
    lattice_tokens = extract_lattice_tokens(struct)
    
    # Extract Wyckoff free parameters
    wyckoff_free_params = extract_wyckoff_free_params(struct, symm_dataset)

    wyckoff_ls = []
    for i in range(len(wyckoff_positions)):
        multiplicity = wyckoff_multiplicity_dict[spg_num][wyckoff_positions[i]]
        wyckoff_symbol = multiplicity + wyckoff_positions[i]
        if wyckoff_symbol not in wyckoff_ls:
            wyckoff_ls.append(wyckoff_symbol)
    
    # New SCOPE+ format: gen_str | LATTICE: lattice_tokens | WYCKOFF: wyckoff_symbols + free_params | composition
    seq = seq + ' | LATTICE: ' + ' '.join(lattice_tokens)
    seq = seq + ' | WYCKOFF: ' + ' '.join(wyckoff_ls) + ' ' + ' '.join(wyckoff_free_params)

    comp_ls = []
    for element, ratio in struct.composition.fractional_composition.get_el_amt_dict().items():
        ratio = str(np.round(ratio, 2))
        comp_ls.append(element)
        comp_ls.append(ratio)
   
    seq = seq + ' | ' + ' '.join(comp_ls)

    return seq

def process_cif(file_path):
    try:
        struct = Structure.from_file(file_path)
        # Replace with your custom conversion function
        string_representation = generate_seq(struct, gen_str, wyckoff_multiplicity_dict)
        return string_representation
    except:
        return None

def save_results(results, output_file):
    df = pd.DataFrame(results, columns=["gen_str"])
    df.to_csv(output_file, mode='a', index=False)
    
    
"""
Functions for full string representation of a structure
"""

def generate_seq_full(struct, gen_str, wyckoff_multiplicity_dict):
    analyzer = SpacegroupAnalyzer(struct)
    symm_dataset = analyzer.get_symmetry_dataset()
    wyckoff_positions = symm_dataset['wyckoffs']

    spg_num = str(analyzer.get_space_group_number())
    seq = " ".join(gen_str[spg_num])

    # Extract lattice parameters
    lattice_tokens = extract_lattice_tokens(struct)
    
    # Extract Wyckoff free parameters
    wyckoff_free_params = extract_wyckoff_free_params(struct, symm_dataset)

    wyckoff_ls = []
    for i in range(len(wyckoff_positions)):
        multiplicity = wyckoff_multiplicity_dict[spg_num][wyckoff_positions[i]]
        wyckoff_symbol = multiplicity + wyckoff_positions[i]
        wyckoff_ls.append(wyckoff_symbol)
    
    # New SCOPE+ format: gen_str | LATTICE: lattice_tokens | WYCKOFF: wyckoff_symbols + free_params | elements
    seq = seq + ' | LATTICE: ' + ' '.join(lattice_tokens)
    seq = seq + ' | WYCKOFF: ' + ' '.join(wyckoff_ls) + ' ' + ' '.join(wyckoff_free_params)

    elements = [site.species_string for site in struct]
    
    assert len(elements) == len(wyckoff_ls)
   
    seq = seq + ' | ' + ' '.join(elements)

    return seq

def process_cif_full(file_path):
    try:
        struct = Structure.from_file(file_path)
        # Replace with your custom conversion function
        string_representation = generate_seq_full(struct, gen_str, wyckoff_multiplicity_dict)
        return string_representation
    except:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get cloud from cif")
    parser.add_argument("--dir", help="Path to the cif file")
    parser.add_argument("--out", help="Path to save cloud csv")
    parser.add_argument("--numproc", help="number of processes")
    parser.add_argument("--batchsize", help="batch size")
    parser.add_argument("--reduce", action="store_true", help="Enable reduced string representation")
    parser.add_argument("--no-reduce", action="store_false", dest="reduce", help="Disable reduced string representation")
    args = parser.parse_args()

    with open("data/wyckoff-position-multiplicities.json") as file:
        # dictionary mapping Wyckoff letters in a given space group to their multiplicity
        wyckoff_multiplicity_dict = json.load(file)

    with open('data/generator.json', 'r') as fp:
        gen_str = json.load(fp)

    cif_files = glob.glob(os.path.join(args.dir, '*.cif'))
    num_batches = len(cif_files) // int(args.batchsize) + 1
    # t1_start = perf_counter() 
    # count = 0
    with Pool(int(args.numproc)) as pool:
        for batch_num in range(num_batches):
            batch_files = cif_files[batch_num * int(args.batchsize):(batch_num + 1) * int(args.batchsize)]
            process_func = process_cif if args.reduce else process_cif_full
            results = pool.map(process_func, batch_files)
            # count += 1
            save_results(results, args.out)
            print(f"Processed batch {batch_num + 1}/{num_batches}")

    # t1_stop = perf_counter()

    # print("Throughput:", count * int(args.batchsize) / (t1_stop - t1_start))

