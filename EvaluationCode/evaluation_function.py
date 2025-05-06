from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd

# CSV files should have these columns: ["RandomID", "Score", "Label"]

# Load your prediction file

def cluster_prc(clusters, all_sel, all_clusters, score = 'Score'):
    cluster_recall = []
    cluster_precision = []

    for th in clusters[score].unique():
        found = clusters.query(f'{score} >= @th').shape[0]
        cluster_recall.append(found/all_clusters)
        selected = all_sel.query(f'{score} >= @th').shape[0]
        cluster_precision.append(found/selected)
    
    return auc(cluster_recall, cluster_precision)


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
    prc = precision_recall_curve(all_data[label_gold], all_data[score])
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

        if n_clusters > 1:
            cluster_prauc = cluster_prc(cluster_team, data_team, all_clusters)
        elif n_clusters == 1:
            th = cluster_team[score].to_list()[-1]
            cluster_prauc = 1/data_team.query(f"{score} >= {th}").shape[0]
        else:
            cluster_prauc = None

        results[f"Clusters_{label_team}"] = n_clusters
        results[f"Hits_{label_team}"] = n_hits
        results[f"ClusterPRAUC_{label_team}"] = cluster_prauc

    return results

#team_df = pd.read_csv("Team.csv")
#gold_df = pd.read_csv("Gold.csv")

#print(evaluate_team_model(gold_df, team_df))