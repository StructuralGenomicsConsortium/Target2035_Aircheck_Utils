# EvaluationCode

**This  script represents the basic framework for evaluation, additional features or changes to the I/O interface might be included in the future**

The script `evaluation_function.py` evaluates the quality of predictions for the DREAM Target 2035 challenge.

The script compares the predictions stored in a `.csv` file (eg. `Team.csv`) or in a pandas dataframe, containing the unique ID of a molecule, its confidence score, and its binary label identifying it as selected for evaluation (1) or not (0), with the gold standard data stored as a `.csv` file (eg. `Gold.csv`) or in a pandas dataframe, containing the binary true label identifying a molecule as a hit (1) or not (0) and a label indicating the cluster containing the selected molecule.

Clustering was performed previously only on the hit molecules in the test set, the algorithm, threshold, and metric used for clustering might vary depending on the test dataset.

The script calculates ROCAUC and PRAUC value on all molecules based on their confidence scores, regardless of the predicted label. Based on the selection label (eg. `Sel_200`) the script determines the number of identified hit molecules, the number of unique clusters, and a pseudo PRAUC value calculated only on unique clusters. The pseudo PRAUC value should only be used to compare the performance of workflows selecting the same number of unique clusters, do not use it to compare selections containing different numbers of unique clusters, since the metric depends on the number of identified clusters.

