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

# Task 2.2 Logistic Regression
w = np.zeros(7)
b = 0.0
eta = 0.1
losses = {}

def bce(X_m, y_m, w_m, b_m):
    z_m = X_m @ w_m + b_m
    p_m = 1.0 / (1.0 + np.exp(-z_m))
    p_m = np.clip(p_m, 1e-15, 1.0 - 1e-15)
    return -np.mean(y_m * np.log(p_m) + (1.0 - y_m) * np.log(1.0 - p_m))

for epoch in range(1001):
    if epoch in [0, 1, 10, 100, 1000]:
        losses[epoch] = bce(X_tr_std, y_train, w, b)
    if epoch < 1000:
        z = X_tr_std @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - y_train
        dw = (1.0 / len(y_train)) * (X_tr_std.T @ err)
        db = np.mean(err)
        w -= eta * dw
        b -= eta * db

test_bce = bce(X_te_std, y_test, w, b)

# Task 2.3 Learning Rate
loss_eta = {}
for e_val in [0.01, 0.1, 1.0]:
    w_temp, b_temp = np.zeros(7), 0.0
    for epoch in range(200):
        z_t = X_tr_std @ w_temp + b_temp
        p_t = 1.0 / (1.0 + np.exp(-z_t))
        err_t = p_t - y_train
        w_temp -= e_val * (1.0 / len(y_train)) * (X_tr_std.T @ err_t)
        b_temp -= e_val * np.mean(err_t)
    loss_eta[e_val] = bce(X_tr_std, y_train, w_temp, b_temp)

# Task 2.4 Odds Ratios
ors = np.exp(w)

# Task 2.5 & 2.6 Evaluate on Test
z_te = X_te_std @ w + b
p_te = 1.0 / (1.0 + np.exp(-z_te))

def eval_thresh(p_arr, y_arr, tau):
    preds = (p_arr >= tau).astype(int)
    tp = int(np.sum((preds == 1) & (y_arr == 1)))
    fp = int(np.sum((preds == 1) & (y_arr == 0)))
    tn = int(np.sum((preds == 0) & (y_arr == 0)))
    fn = int(np.sum((preds == 0) & (y_arr == 1)))
    return tp, fp, tn, fn

blind_acc = np.mean(0 == y_test)

with open("outputs/part2_thresholds.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])
    for tau_val in np.round(np.arange(0.1, 1.0, 0.1), 1):
        tp, fp, tn, fn = eval_thresh(p_te, y_test, tau_val)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        writer.writerow([f"{tau_val:.1f}", tp, fp, tn, fn, f"{tpr:.4f}", f"{fpr:.4f}"])

# Task 2.7 KNN CV
best_k = 1
max_cv_f1 = -1.0
cv_results = []

for k in range(1, 23, 2):
    f1_folds = []
    for fold in range(5):
        val_mask = (np.arange(len(y_train)) % 5) == fold
        train_mask = ~val_mask
        
        X_ref, y_ref = X_tr_std[train_mask], y_train[train_mask]
        X_val, y_val = X_tr_std[val_mask], y_train[val_mask]
        
        y_val_preds = []
        for q in X_val:
            dists = np.sum((X_ref - q)**2, axis=1)
            sort_k = np.argsort(dists, kind='stable')[:k]
            pos_cnt = np.sum(y_ref[sort_k])
            y_val_preds.append(1 if pos_cnt >= (k + 1) // 2 else 0)
            
        y_val_preds = np.array(y_val_preds)
        tp = np.sum((y_val_preds == 1) & (y_val == 1))
        fp = np.sum((y_val_preds == 1) & (y_val == 0))
        fn = np.sum((y_val_preds == 0) & (y_val == 1))
        _, _, _, _, _, f1_val = compute_metrics(tp, fp, int(np.sum((y_val_preds == 0) & (y_val == 0))), fn)
        f1_folds.append(f1_val)
        
    mean_f1 = np.mean(f1_folds)
    cv_results.append((k, mean_f1))
    if mean_f1 > max_cv_f1:
        max_cv_f1 = mean_f1
        best_k = k

with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "mean_cv_f1"])
    for k, mean_f1 in cv_results:
        writer.writerow([k, f"{mean_f1:.4f}"])

knn_preds = []
for q in X_te_std:
    dists = np.sum((X_tr_std - q)**2, axis=1)
    sort_k = np.argsort(dists, kind='stable')[:best_k]
    pos_cnt = np.sum(y_train[sort_k])
    knn_preds.append(1 if pos_cnt >= (best_k + 1) // 2 else 0)

knn_preds = np.array(knn_preds)
knn_tp = int(np.sum((knn_preds == 1) & (y_test == 1)))
knn_fp = int(np.sum((knn_preds == 1) & (y_test == 0)))
knn_tn = int(np.sum((knn_preds == 0) & (y_test == 0)))
knn_fn = int(np.sum((knn_preds == 0) & (y_test == 1)))

# Task 2.8 Money Threshold
z_tr = X_tr_std @ w + b
p_tr = 1.0 / (1.0 + np.exp(-z_tr))

min_cost = float('inf')
cost_tau = 0.0

for k in range(1, 20):
    tau_cand = k / 20.0
    preds_cand = (p_tr >= tau_cand).astype(int)
    fp_cand = np.sum((preds_cand == 1) & (y_train == 0))
    fn_cand = np.sum((preds_cand == 0) & (y_train == 1))
    cost_cand = 50 * fp_cand + 400 * fn_cand
    if cost_cand < min_cost:
        min_cost = cost_cand
        cost_tau = tau_cand

cost_train_at_tau = min_cost

test_preds_tau = (p_te >= cost_tau).astype(int)
cost_test_at_tau = 50 * np.sum((test_preds_tau == 1) & (y_test == 0)) + 400 * np.sum((test_preds_tau == 0) & (y_test == 1))

test_preds_05 = (p_te >= 0.5).astype(int)
cost_test_at_05 = 50 * np.sum((test_preds_05 == 1) & (y_test == 0)) + 400 * np.sum((test_preds_05 == 0) & (y_test == 1))

cost_test_flag_nobody = 400 * np.sum(y_test == 1)

with open("outputs/part2_results.txt", "w") as f:
    f.write(f"train_rows = {len(X_train)}\n")
    f.write(f"train_positives = {np.sum(y_train)}\n")
    f.write(f"test_rows = {len(X_test)}\n")
    f.write(f"test_positives = {np.sum(y_test)}\n")
    f.write(f"loss_epoch_0 = {losses[0]:.4f}\n")
    f.write(f"loss_epoch_1 = {losses[1]:.4f}\n")
    f.write(f"loss_epoch_10 = {losses[10]:.4f}\n")
    f.write(f"loss_epoch_100 = {losses[100]:.4f}\n")
    f.write(f"loss_epoch_1000 = {losses[1000]:.4f}\n")
    for i in range(7):
        f.write(f"weight_X{i+1} = {w[i]:.4f}\n")
    f.write(f"bias = {b:.4f}\n")
    f.write(f"test_bce = {test_bce:.4f}\n")
    f.write(f"loss_epoch_200_eta_0.01 = {loss_eta[0.01]:.4f}\n")
    f.write(f"loss_epoch_200_eta_0.1 = {loss_eta[0.1]:.4f}\n")
    f.write(f"loss_epoch_200_eta_1.0 = {loss_eta[1.0]:.4f}\n")
    for i in range(7):
        f.write(f"odds_ratio_X{i+1} = {ors[i]:.4f}\n")
    
    for tau_v in [0.5, 0.3]:
        tp, fp, tn, fn = eval_thresh(p_te, y_test, tau_v)
        acc, prec, rec, spec, _, f1 = compute_metrics(tp, fp, tn, fn)
        prefix = f"tau_{tau_v:.1f}"
        f.write(f"{prefix}_tp = {tp}\n{prefix}_fp = {fp}\n{prefix}_tn = {tn}\n{prefix}_fn = {fn}\n")
        f.write(f"{prefix}_accuracy = {acc:.4f}\n{prefix}_precision = {prec:.4f}\n{prefix}_recall = {rec:.4f}\n{prefix}_f1 = {f1:.4f}\n{prefix}_specificity = {spec:.4f}\n")
        
    f.write(f"blind_accuracy = {blind_acc:.4f}\n")
    f.write(f"knn_best_k = {best_k}\n")
    f.write(f"knn_tp = {knn_tp}\nknn_fp = {knn_fp}\nknn_tn = {knn_tn}\nknn_fn = {knn_fn}\n")
    acc_k, prec_k, rec_k, _, _, f1_k = compute_metrics(knn_tp, knn_fp, knn_tn, knn_fn)
    f.write(f"knn_accuracy = {acc_k:.4f}\nknn_precision = {prec_k:.4f}\nknn_recall = {rec_k:.4f}\nknn_f1 = {f1_k:.4f}\n")
    f.write(f"cost_tau = {cost_tau:.4f}\n")
    f.write(f"cost_train_at_tau = {cost_train_at_tau}\n")
    f.write(f"cost_test_at_tau = {cost_test_at_tau}\n")
    f.write(f"cost_test_at_0.5 = {cost_test_at_05}\n")
    f.write(f"cost_test_flag_nobody = {cost_test_flag_nobody}\n")