import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from fingerprints import HitGenMACCS, HitGenECFP4, HitGenECFP6, HitGenFCFP4, HitGenFCFP6, HitGenRDK, HitGenAvalon, HitGenTopTor, HitGenAtomPair

def compute_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        alogp = Descriptors.MolLogP(mol)
    else:
        mw = np.nan
        alogp = np.nan
    return mw, alogp

def generate_fingerprints(smiles, fps_dict):
    fp_data = {}
    for fp_name, fp_class in fps_dict.items():
        try:
            fp_array = fp_class.generate_fps(smis=[smiles]).flatten()
            fp_data[fp_name] = ','.join(map(str, fp_array))
        except Exception:
            fp_data[fp_name] = ','.join(['nan'] * fp_class._dimension)
    return fp_data

def process_file(input_file, output_file, fingerprints):
    # Read the parquet file
    df = pd.read_parquet(input_file, engine='pyarrow')

    # Generate fingerprint columns
    fingerprint_data = []
    k = 0
    for smiles in df['SMILES']:
        k = k + 1
        print(k)
        fps = generate_fingerprints(smiles, fingerprints)
        fingerprint_data.append(fps)

    # Create a DataFrame from fingerprint data
    fingerprint_df = pd.DataFrame(fingerprint_data)

    # Concatenate fingerprint data with the main DataFrame
    df = pd.concat([df, fingerprint_df], axis=1)

    # Save the new DataFrame to a parquet file
    df.to_parquet(output_file, engine='pyarrow', index=False)

    print(f"The updated file with fingerprints has been saved as '{output_file}'")

def main():
    # Define fingerprint classes
    fingerprint_classes = {
        'ECFP4': HitGenECFP4(),
        'ECFP6': HitGenECFP6(),
        'FCFP4': HitGenFCFP4(),
        'FCFP6': HitGenFCFP6(),
        'MACCS': HitGenMACCS(),
        'RDK': HitGenRDK(),
        'AVALON': HitGenAvalon(),
        'TOPTOR': HitGenTopTor(),
        'ATOMPAIR': HitGenAtomPair()
    }

    input_file = "Data.parquet"  # Parquet file
    output_file = "Data_with_fingerprints.parquet"  # Parquet file
    process_file(input_file, output_file, fingerprint_classes)

if __name__ == "__main__":
    main()
