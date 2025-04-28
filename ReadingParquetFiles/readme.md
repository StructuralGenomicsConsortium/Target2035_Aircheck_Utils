# Read Parquet Utilities

This utility provides simple functions to read `.parquet` files and process fingerprint columns.

## Functions

- **read_parquet_file(file_path, columns=None, nrows=None)**  
  Reads a parquet file, optionally selecting specific columns and limiting rows.

- **process_column_to_array(df, column_name)**  
  Converts a fingerprint column from a comma-separated string to a NumPy array (if needed).

## Usage Examples

```python
from read_parquet_utils import read_parquet_file, process_column_to_array

# Sample 1: Read array-format fingerprints
df = read_parquet_file('Data-arrayFP.parquet', columns=["ECFP4"], nrows=50)
X = process_column_to_array(df, "ECFP4")

# Sample 2: Read string-format fingerprints
df = read_parquet_file('Data-stringFP.parquet', columns=["ECFP4"], nrows=50)
X = process_column_to_array(df, "ECFP4")
