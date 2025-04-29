# **Target2035_Aircheck_Utils**

## Fingerprint Extraction

This folder provides utilities for extracting molecular fingerprints from SMILES strings stored in a Parquet file.  
It supports several fingerprint types: ECFP4, ECFP6, FCFP4, FCFP6, MACCS, RDK, Avalon, TopTor, and AtomPair.

### Main Script

- **ExtractingFingerprints.py**  
  - Reads an input `.parquet` file containing a `SMILES` column.
  - Computes selected fingerprints for each molecule.
  - Adds the fingerprint vectors as new columns.
  - Saves the result to a new `.parquet` file.

### Input

- A Parquet file (e.g., `Data.parquet`) with at least a `SMILES` column.

### Output

- A new Parquet file (e.g., `Data_with_fingerprints.parquet`) containing:
  - Original data
  - Additional columns for each fingerprint type.

### How to Use

1. Place your input Parquet file in the folder.
2. Run `ExtractingFingerprints.py`.
3. The output file with fingerprints will be saved automatically.

### Requirements

- `pandas`
- `numpy`
- `rdkit`
- `pyarrow`

