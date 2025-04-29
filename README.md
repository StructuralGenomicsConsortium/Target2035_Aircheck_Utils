# Target2035 Utilities Repository

This repository provides essential tools and example workflows to support participants in the Target2035 challenge. It includes utilities for molecular fingerprint extraction, data loading, model training, and evaluation.

## Contents

- **Fingerprint Extraction**  
  Scripts to compute molecular fingerprints (ECFP4, ECFP6, MACCS, etc.) from SMILES and save them into a Parquet file.  
  📁 folder name: FingerprintExtraction

- **Parquet Reader Utilities**  
  Simple tools to load Parquet files and process fingerprint columns into NumPy arrays.  
  📁 folder name: ReadingParquetFiles

- **Training Notebook**  
  A notebook covering training set loading, cross-validation, model training, and optional clustering of results.  
  📁 folder name: Notebooks

- **Evaluation Code**  
  Scripts that accept the gold label files and participants' predicted outputs, and compute evaluation metrics.  
  📁 folder name: EvaluationCode
