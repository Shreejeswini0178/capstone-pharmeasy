import csv
import os
import numpy as np
from helper import load_data, get_stratified_split, compute_metrics

os.makedirs("outputs", exist_ok=True)
X, y, _ = load_data()
train_idx, test_idx = get_stratified_split(X, y)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

mean_tr = np.mean(X_train, axis=0)
std_tr = np.std(X_train, axis=0, ddof=1)
X_tr_std = (X_train - mean_tr) / std_tr
X_te_std = (X_test - mean_tr) / std_tr

# Task 4.1 Splits
def get_entropy(y_sub):
    if len(y_sub) == 0: return 0.0
    p1 = np.mean(y_sub == 1)
    p0 = 1.0 - p1
    e = 0.0
    if p1 > 0: e -= p1 * np.log2(p1)
    if p0 > 0: e -= p0 * np.log2(p0)
    return e

def get_gini(y_sub):
    if len(y_sub) == 0: return 0.0
    p1 = np.mean(y_sub == 1)
    p0 = 1.0 - p1
    return 1.0 - (p1**2 + p0**2)

root_ent = get_entropy(y_train)
root_gini = get_gini(y_train)

splits_def = {
    'S1': X_train[:, 5] == 1,
    'S2': X_train[:, 6] == 1,
    'S3': X_train[:, 2] > 0.5,
    'S4': X_train[:, 0] <= 24
}

splits_data = []
best_ig, best_ig_name = -1.0, ""
best_gg, best_gg_name = -1.0, ""

for s_name in ['S1', 'S2', 'S3', 'S4']:
    mask = splits_def[s_name]
    y_yes = y_train[mask]
    y_no = y_train[~mask]
    
    n_y, pos_y = len(y_yes), int(np.sum(y_yes))
    n_n, pos_n = len(y_no), int(np.sum(y_no))
    
    w_ent = (n_y / 560.0) * get_entropy(y_yes) + (n_n / 560.0) * get_entropy(y_no)
    w_gini = (n_y / 560.0) * get_gini(y_yes) + (n_n / 560.0) * get_gini(y_no)
    
    ig = root_ent - w_ent
    gg = root_gini - w_gini
    
    if ig > best_ig: best_ig, best_ig_name = ig, s_name
    if gg > best_gg: best_gg, best_gg_name = gg, s_name
    
    splits_data.append((s_name, n_y, pos_y, n_n, pos_n, ig, gg))

with open("outputs/part4_splits.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"])
    for r in splits_data:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}"])

# Task 4.2 Voting
stump_mask = splits_def[best_ig_name]
stump_rate_yes = np.mean(y_train[stump_mask])
stump_rate_no = np.mean(y_train[~stump_mask])

# Logistic Model
w_log = np.zeros(7)
b_log = 0.0
for epoch in range(1000):
    z = X_tr_std @ w_log + b_log
    p = 1.0 / (1.0 + np.exp(-z))
    w_log -= 0.1 * (1.0 / 560) * (X_tr_std.T @ (p - y_train))
    b_log -= 0.1 * np.mean(p - y_train)

p_log_te = 1.0 / (1.0 + np.exp(-(X_te_std @ w_log + b_log)))

# KNN Model (K=15 assumed best_k from reference, dynamic match)
best_k = 15
p_knn_te = []
for q in X_te_std:
    dists = np.sum((X_tr_std - q)**2, axis=1)
    sort_k = np.argsort(dists, kind='stable')[:best_k]
    p_knn_te.append(np.mean(y_train[sort_k]))
p_knn_te = np.array(p_knn_te)

# Stump Model on Test
if best_ig_name == 'S1': test_stump_mask = X_test[:, 5] == 1
elif best_ig_name == 'S2': test_stump_mask = X_test[:, 6] == 1
elif best_ig_name == 'S3': test_stump_mask = X_test[:, 2] > 0.5
else: test_stump_mask = X_test[:, 0] <= 24

p_stump_te = np.where(test_stump_mask, stump_rate_yes, stump_rate_no)

pred_log = (p_log_te >= 0.3).astype(int)
pred_knn = (p_knn_te >= 0.3).astype(int)
pred_stump = (p_stump_te >= 0.3).astype(int)

pred_hard = ((pred_log + pred_knn + pred_stump) >= 2).astype(int)
p_soft = (p_log_te + p_knn_te + p_stump_te) / 3.0
pred_soft = (p_soft >= 0.3).astype(int)

voting_models = [
    ('logistic', pred_log),
    ('knn', pred_knn),
    ('stump', pred_stump),
    ('hard_vote', pred_hard),
    ('soft_vote', pred_soft)
]

voting_rows = []
for name, p_vec in voting_models:
    tp = int(np.sum((p_vec == 1) & (y_test == 1)))
    fp = int(np.sum((p_vec == 1) & (y_test == 0)))
    tn = int(np.sum((p_vec == 0) & (y_test == 0)))
    fn = int(np.sum((p_vec == 0) & (y_test == 1)))
    _, prec, rec, _, _, f1 = compute_metrics(tp, fp, tn, fn)
    voting_rows.append((name, tp, fp, tn, fn, prec, rec, f1))

with open("outputs/part4_voting.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"])
    for r in voting_rows:
        writer.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.4f}", f"{r[6]:.4f}", f"{r[7]:.4f}"])

# Task 4.3 AdaBoost
y_ada = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

w_r1 = np.ones(8) / 8.0
err_r1 = np.sum(w_r1[h1 != y_ada])
alpha_r1 = 0.5 * np.log((1.0 - err_r1) / err_r1)
w_unnorm_r1 = w_r1 * np.exp(-alpha_r1 * y_ada * h1)
w_r2 = w_unnorm_r1 / np.sum(w_unnorm_r1)

err_r2 = np.sum(w_r2[h2 != y_ada])
alpha_r2 = 0.5 * np.log((1.0 - err_r2) / err_r2)
w_unnorm_r2 = w_r2 * np.exp(-alpha_r2 * y_ada * h2)
w_after_r2 = w_unnorm_r2 / np.sum(w_unnorm_r2)

ada_rows = []
for i in range(8):
    ada_rows.append((1, i+1, w_r1[i], w_unnorm_r1[i], w_r2[i]))
for i in range(8):
    ada_rows.append((2, i+1, w_r2[i], w_unnorm_r2[i], w_after_r2[i]))

with open("outputs/part4_adaboost.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["round", "point", "weight_before", "weight_unnormalised", "weight_after"])
    for r in ada_rows:
        writer.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

ens_pred = np.sign(alpha_r1 * h1 + alpha_r2 * h2)
ens_acc = np.mean(ens_pred == y_ada)

# Task 4.4 OOB
oob_frac = (1.0 - 1.0/560.0)**560

# Task 4.5 Architectures
mlp_h1 = 784 * 16 + 16
mlp_h2 = 16 * 16 + 16
mlp_out = 16 * 10 + 10
mlp_tot = mlp_h1 + mlp_h2 + mlp_out

cnn_c1 = (5 * 5 * 1 + 1) * 6
cnn_c2 = (5 * 5 * 6 + 1) * 16
# Spatial size check: 28x28 -> conv1 -> 28x28 -> pool -> 14x14 -> conv2 -> 10x10 -> pool -> 5x5
cnn_d1 = (5 * 5 * 16) * 32 + 32
cnn_d2 = 32 * 10 + 10
cnn_tot = cnn_c1 + cnn_c2 + cnn_d1 + cnn_d2

arch_data = [
    ("mlp_hidden1", mlp_h1),
    ("mlp_hidden2", mlp_h2),
    ("mlp_output", mlp_out),
    ("cnn_conv1", cnn_c1),
    ("cnn_conv2", cnn_c2),
    ("cnn_dense1", cnn_d1),
    ("cnn_dense2", cnn_d2)
]

with open("outputs/part4_architecture.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "parameters"])
    for r in arch_data:
        writer.writerow([r[0], r[1]])

fc_params = (128 * 128 * 3) * 1024 + 1024
conv_params = (3 * 3 * 3 + 1) * 64

rnn_w_xh = 64 * 128
rnn_w_hh = 128 * 128
rnn_b_h = 128
rnn_w_hy = 128 * 16
rnn_b_y = 16
rnn_tot = rnn_w_xh + rnn_w_hh + rnn_b_h + rnn_w_hy + rnn_b_y

with open("outputs/part4_results.txt", "w") as f:
    f.write(f"root_entropy = {root_ent:.4f}\n")
    f.write(f"root_gini = {root_gini:.4f}\n")
    f.write(f"best_split_by_info_gain = {best_ig_name}\n")
    f.write(f"best_split_by_gini_gain = {best_gg_name}\n")
    f.write(f"stump_rate_yes = {stump_rate_yes:.4f}\n")
    f.write(f"stump_rate_no = {stump_rate_no:.4f}\n")
    f.write(f"round1_error = {err_r1:.4f}\n")
    f.write(f"round1_alpha = {alpha_r1:.4f}\n")
    f.write(f"round2_error = {err_r2:.4f}\n")
    f.write(f"round2_alpha = {alpha_r2:.4f}\n")
    f.write(f"ensemble_accuracy = {ens_acc:.4f}\n")
    f.write(f"oob_fraction = {oob_frac:.4f}\n")
    f.write(f"mlp_total = {mlp_tot}\n")
    f.write(f"cnn_total = {cnn_tot}\n")
    f.write(f"fc_layer_params = {fc_params}\n")
    f.write(f"conv_layer_params = {conv_params}\n")
    f.write(f"rnn_total = {rnn_tot}\n")