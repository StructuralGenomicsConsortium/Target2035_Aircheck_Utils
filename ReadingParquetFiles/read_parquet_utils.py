import pandas as pd
import numpy as np

def read_parquet_file(file_path, columns=None, nrows=None):
    df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    if nrows is not None:
        df = df.head(nrows)
    return df

def process_column_to_array(df, column_name):
    if isinstance(df[column_name].iloc[0], str):
        # Column is string, needs conversion
        return np.stack(df[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
    else:
        # Column is already array-like
        return np.stack(df[column_name])
