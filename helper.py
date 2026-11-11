import csv
import numpy as np

def load_data():
    X, y, val = [], [], []
    with open("data/records.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            X.append([float(x) for x in r[1:8]])
            y.append(int(r[8]))
            val.append(float(r[9]))
    return np.array(X), np.array(y), np.array(val)

def get_stratified_split(X, y):
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    n_train_0 = (7 * len(idx_0)) // 10
    n_train_1 = (7 * len(idx_1)) // 10
    
    train_idx = np.sort(np.concatenate([idx_0[:n_train_0], idx_1[:n_train_1]]))
    test_idx = np.sort(np.concatenate([idx_0[n_train_0:], idx_1[n_train_1:]]))
    
    return train_idx, test_idx

def compute_metrics(tp, fp, tn, fn):
    tot = tp + fp + tn + fn
    acc = (tp + tn) / tot if tot > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, spec, fpr, f1