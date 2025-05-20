from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd
from scipy.stats import poisson_binom
import numpy as np

# CSV files should have these columns: ["RandomID", "Score", "Label"]

# Load your prediction file

def cluster_prc(clusters, all_sel, all_clusters, score = 'Score'):
    cluster_recall = [0]
    cluster_precision = [0]

    for th in clusters[score].unique():
        found = clusters.query(f'{score} >= @th').shape[0]
        cluster_recall.append(found/all_clusters)
        selected = all_sel.query(f'{score} >= @th').shape[0]
        cluster_precision.append(found/selected)
    
    return auc(cluster_recall, cluster_precision)/auc(cluster_recall, [0] + [1 for _ in cluster_precision[:-1]])

def cluster_hits_p(gold_df, K, M, cluster = 'Cluster'):
    clust_labels = gold_df[cluster].dropna().to_numpy()
    probs = []
    n_clusters = len(set(clust_labels))
    probs = np.stack(
        [(clust_labels == p).sum()/gold_df.shape[0] for p in set(clust_labels)]
        )
    p_hits = 1 - (1-probs)**M
    if K < n_clusters/2:
        return 1-np.sum(
            [poisson_binom.pmf(i, p_hits) for i in range(K)]
            )
    else:
        return np.sum(
            [poisson_binom.pmf(i, p_hits) for i in range(K, n_clusters+1)]
            )


def evaluate_team_model(gold_df, team_df,
                        label_gold = 'Label',
                        score = 'Score',
                        labels_team = ["Sel_200", "Sel_500"],
                        cluster = 'Cluster',
                        random_id = 'RandomID'):

    results = {}

    # Merge on RandomID
    all_data = pd.merge(gold_df, team_df, on=random_id, suffixes=('_gold', '_team'))
    
    # Compute metrics all data
    rocauc = roc_auc_score(all_data[label_gold], all_data[score])
    prc = precision_recall_curve(y_true=all_data[label_gold], y_score=all_data[score])
    prauc = auc(prc[1], prc[0])

    results['ROCAUC'] = rocauc
    results['PRAUC'] = prauc

    all_clusters = gold_df.query(f"{label_gold} == 1").drop_duplicates(cluster).shape[0]
        
    # Calculate metrics on selections
    for label_team in labels_team:
        data_team = all_data.query(f"{label_team} == 1")
        hits_team = data_team.query(f"{label_gold} == 1")
        cluster_team = hits_team.sort_values(score, ascending = False).drop_duplicates(cluster)
        n_hits = hits_team.shape[0]
        n_clusters = cluster_team.shape[0]

        if n_clusters == 0:
            cluster_prauc = None
            p_value = None
        else:
            p_value = cluster_hits_p(gold_df, n_clusters, all_data[label_team].sum())
            if len(data_team[score].unique()) == 1:
                cluster_prauc = None
            elif n_clusters > 1:
                cluster_prauc = cluster_prc(cluster_team, data_team, all_clusters)
            else:
                th = cluster_team[score].to_list()[-1]
                cluster_prauc = 1/data_team.query(f"{score} >= {th}").shape[0]   

        results[f"Clusters_{label_team}"] = n_clusters
        results[f"Hits_{label_team}"] = n_hits
        results[f"ClusterPRAUC_{label_team}"] = cluster_prauc
        results[f"P-value_{label_team}"] = p_value

    return results

#team_df = pd.read_csv("Team.csv")
#gold_df = pd.read_csv("Gold.csv")

#print(evaluate_team_model(gold_df, team_df))