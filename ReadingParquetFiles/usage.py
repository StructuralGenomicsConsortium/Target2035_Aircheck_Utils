from read_parquet_utils import read_parquet_file, process_column_to_array


# Sample 1:
# Load specific column and limit rows
df = read_parquet_file('Data-arrayFP.parquet', columns=["ECFP4"], nrows=50)
# Process if needed
X = process_column_to_array(df, "ECFP4")


# Sample 2: 
df = read_parquet_file('Data-stringFP.parquet', columns=["ECFP4"], nrows=50)
X = process_column_to_array(df, "ECFP4")

