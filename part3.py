import csv
import os
import numpy as np
from helper import load_data

os.makedirs("outputs", exist_ok=True)
X_all, y_all, val_all = load_data()
X_5 = X_all[:, :5]

mean_5 = np.mean(X_5, axis=0)
std_5 = np.std(X_5, axis=0, ddof=1)
X_std = (X_5 - mean_5) / std_5

# Task 3.1 K-Means
wcss_dict = {}
k_centroids = {}
k_assignments = {}

for K in range(2, 7):
    init_indices = [(i * (800 // K)) for i in range(K)]
    centroids = X_std[init_indices].copy()
    
    while True:
        dists = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        assignments = np.argmin(dists, axis=1)
        
        new_centroids = np.copy(centroids)
        for k in range(K):
            members = X_std[assignments == k]
            if len(members) > 0:
                new_centroids[k] = np.mean(members, axis=0)
                
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
        
    wcss = np.sum((X_std - centroids[assignments])**2)
    wcss_dict[K] = wcss
    k_centroids[K] = centroids
    k_assignments[K] = assignments

# Task 3.2 Elbow
elbow_k = 3
for K in [3, 4, 5]:
    drop1 = wcss_dict[K] - wcss_dict[K+1]
    drop2 = wcss_dict[K-1] - wcss_dict[K]
    if drop1 < 0.5 * drop2:
        elbow_k = K
        break

# Task 3.3 Profile K=3 Segments
assign_3 = k_assignments[3]
seg_X3_means = [np.mean(X_5[assign_3 == k, 2]) for k in range(3)]
order = np.argsort(seg_X3_means)

seg_map = {order[0]: 'A', order[1]: 'B', order[2]: 'C'}
named_assign = np.array([seg_map[a] for a in assign_3])

seg_data = []
for label in ['A', 'B', 'C']:
    mask = named_assign == label
    size = int(np.sum(mask))
    m_X = np.mean(X_5[mask], axis=0)
    out_rate = np.mean(y_all[mask])
    m_val = np.mean(val_all[mask])
    seg_data.append((label, size, m_X[0], m_X[1], m_X[2], m_X[3], m_X[4], out_rate, m_val))

with open("outputs/part3_segments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"])
    for row in seg_data:
        writer.writerow([row[0], row[1], f"{row[2]:.4f}", f"{row[3]:.4f}", f"{row[4]:.4f}", f"{row[5]:.4f}", f"{row[6]:.4f}", f"{row[7]:.4f}", f"{row[8]:.4f}"])

# Task 3.4 Hierarchical Clustering
E_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

def run_agglomerative(linkage_type):
    clusters = [[f"E{i+1}"] for i in range(6)]
    merges = []
    
    for step in range(1, 6):
        min_d = float('inf')
        best_pair = (-1, -1)
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                c1, c2 = clusters[i], clusters[j]
                dists = [E_matrix[int(e1[1])-1, int(e2[1])-1] for e1 in c1 for e2 in c2]
                if linkage_type == 'single':
                    d = min(dists)
                elif linkage_type == 'complete':
                    d = max(dists)
                else:
                    d = np.mean(dists)
                if d < min_d:
                    min_d = d
                    best_pair = (i, j)
                    
        i, j = best_pair
        c1, c2 = clusters[i], clusters[j]
        new_c = sorted(c1 + c2, key=lambda x: int(x[1]))
        
        min_m1 = min([int(x[1]) for x in c1])
        min_m2 = min([int(x[1]) for x in c2])
        
        if min_m1 < min_m2:
            ca_str = "+".join(sorted(c1, key=lambda x: int(x[1])))
            cb_str = "+".join(sorted(c2, key=lambda x: int(x[1])))
        else:
            ca_str = "+".join(sorted(c2, key=lambda x: int(x[1])))
            cb_str = "+".join(sorted(c1, key=lambda x: int(x[1])))
            
        merges.append((step, ca_str, cb_str, min_d))
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_c)
        
    return merges

hier_rows = []
cut_results = {}
for link in ['single', 'complete', 'average']:
    m_list = run_agglomerative(link)
    for m in m_list:
        hier_rows.append((link, m[0], m[1], m[2], m[3]))
        
    heights = [m[3] for m in m_list]
    gaps = [heights[i] - heights[i-1] for i in range(1, len(heights))]
    max_gap_idx = np.argmax(gaps)
    num_merges_below = max_gap_idx + 1
    cut_results[link] = 6 - num_merges_below

with open("outputs/part3_hierarchical.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["linkage", "merge_no", "cluster_a", "cluster_b", "height"])
    for r in hier_rows:
        writer.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.4f}"])

# Task 3.5 PCA
S = (1.0 / (800 - 1)) * (X_std.T @ X_std)
eigenvalues, eigenvectors = np.linalg.eigh(S)

idx_sort = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx_sort]
eigenvectors = eigenvectors[:, idx_sort]

for col in range(5):
    max_abs_idx = np.argmax(np.abs(eigenvectors[:, col]))
    if eigenvectors[max_abs_idx, col] < 0:
        eigenvectors[:, col] *= -1.0

eval_sum = np.sum(eigenvalues)
exp_ratio = eigenvalues / eval_sum
cum_ratio = np.cumsum(exp_ratio)
comp_80 = int(np.argmax(cum_ratio >= 0.8) + 1)

W2 = eigenvectors[:, :2]
Z = X_std @ W2

mean_pc1_A = np.mean(Z[named_assign == 'A', 0])
mean_pc1_B = np.mean(Z[named_assign == 'B', 0])
mean_pc1_C = np.mean(Z[named_assign == 'C', 0])

with open("outputs/part3_results.txt", "w") as f:
    for k in range(2, 7):
        f.write(f"wcss_k{k} = {wcss_dict[k]:.4f}\n")
    f.write(f"elbow_k = {elbow_k}\n")
    for row in seg_data:
        f.write(f"size_segment_{row[0]} = {row[1]}\n")
    f.write(f"single_clusters_at_cut = {cut_results['single']}\n")
    f.write(f"complete_clusters_at_cut = {cut_results['complete']}\n")
    f.write(f"average_clusters_at_cut = {cut_results['average']}\n")
    f.write(f"eigenvalue_sum = {eval_sum:.4f}\n")
    for i in range(5):
        f.write(f"eigenvalue_{i+1} = {eigenvalues[i]:.4f}\n")
    for i in range(5):
        f.write(f"explained_ratio_{i+1} = {exp_ratio[i]:.4f}\n")
    for i in range(5):
        f.write(f"cumulative_ratio_{i+1} = {cum_ratio[i]:.4f}\n")
    f.write(f"components_for_80pct = {comp_80}\n")
    for i in range(5):
        f.write(f"pc1_loading_X{i+1} = {eigenvectors[i, 0]:.4f}\n")
    for i in range(5):
        f.write(f"pc2_loading_X{i+1} = {eigenvectors[i, 1]:.4f}\n")
    f.write(f"row1_pc1 = {Z[0, 0]:.4f}\nrow1_pc2 = {Z[0, 1]:.4f}\n")
    f.write(f"row2_pc1 = {Z[1, 0]:.4f}\nrow2_pc2 = {Z[1, 1]:.4f}\n")
    f.write(f"row3_pc1 = {Z[2, 0]:.4f}\nrow3_pc2 = {Z[2, 1]:.4f}\n")
    f.write(f"mean_pc1_segment_A = {mean_pc1_A:.4f}\n")
    f.write(f"mean_pc1_segment_B = {mean_pc1_B:.4f}\n")
    f.write(f"mean_pc1_segment_C = {mean_pc1_C:.4f}\n")