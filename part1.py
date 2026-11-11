import os
import numpy as np
from helper import load_data

os.makedirs("outputs", exist_ok=True)
X, y, value = load_data()
n = len(value)

# Task 1.1
mean_val = np.mean(value)
sd_val = np.std(value, ddof=1)
se_val = sd_val / np.sqrt(n)
ci_low = mean_val - 1.96 * se_val
ci_high = mean_val + 1.96 * se_val

# Task 1.2
z_stat = (mean_val - 8.0) / se_val
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

# Task 1.3
E = 0.5
n_req = int(np.ceil((1.96 * sd_val / E)**2))

# Task 1.4
X4 = X[:, 3]
mean_X4 = np.mean(X4)
cov_X4_v = np.sum((X4 - mean_X4) * (value - mean_val)) / (n - 1)
var_X4 = np.var(X4, ddof=1)
reg_slope = cov_X4_v / var_X4
reg_intercept = mean_val - reg_slope * mean_X4

fitted = reg_intercept + reg_slope * X4
tss = np.sum((value - mean_val)**2)
ess = np.sum((fitted - mean_val)**2)
rss = np.sum((value - fitted)**2)
tss_eq = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
r2 = 1.0 - (rss / tss)
rmse = np.sqrt(rss / n)
mae = np.mean(np.abs(value - fitted))

# Task 1.5
z = (X4 - mean_X4) / np.std(X4, ddof=1)
m, c = 0.0, 0.0
eta = 0.1
gd_costs = {}
for i in range(501):
    e = value - (m * z + c)
    cost = np.mean(e**2)
    if i in [0, 1, 10, 100, 500]:
        gd_costs[i] = cost
    if i < 500:
        dm = -2.0 * np.mean(z * e)
        dc = -2.0 * np.mean(e)
        m -= eta * dm
        c -= eta * dc

gd_slope = m / np.std(X4, ddof=1)
gd_intercept = c - m * mean_X4 / np.std(X4, ddof=1)
gd_slope_gap = abs(gd_slope - reg_slope)

# Task 1.6
sort_idx = np.argsort(fitted, kind='stable')
lower_idx, upper_idx = sort_idx[:400], sort_idx[400:]
resid_raw = value - fitted
resid_sd_lower_raw = np.std(resid_raw[lower_idx], ddof=1)
resid_sd_upper_raw = np.std(resid_raw[upper_idx], ddof=1)
resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

ln_v = np.log(value)
ln_X4 = np.log(X4)
mean_ln_v, mean_ln_X4 = np.mean(ln_v), np.mean(ln_X4)
cov_log = np.sum((ln_X4 - mean_ln_X4) * (ln_v - mean_ln_v)) / (n - 1)
var_log = np.var(ln_X4, ddof=1)
loglog_slope = cov_log / var_log
loglog_intercept = mean_ln_v - loglog_slope * mean_ln_X4

fitted_log = loglog_intercept + loglog_slope * ln_X4
tss_log = np.sum((ln_v - mean_ln_v)**2)
rss_log = np.sum((ln_v - fitted_log)**2)
loglog_r2 = 1.0 - (rss_log / tss_log)
loglog_rmse = np.sqrt(rss_log / n)

sort_idx_log = np.argsort(fitted_log, kind='stable')
resid_log = ln_v - fitted_log
resid_sd_lower_log = np.std(resid_log[sort_idx_log[:400]], ddof=1)
resid_sd_upper_log = np.std(resid_log[sort_idx_log[400:]], ddof=1)
resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

# Task 1.7
X_mat = np.column_stack([np.ones(n), ln_X4, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(X_mat.T @ X_mat, X_mat.T @ ln_v)

fitted_mr = X_mat @ beta
rss_mr = np.sum((ln_v - fitted_mr)**2)
mr_r2 = 1.0 - (rss_mr / tss_log)
mr_adj_r2 = 1.0 - (1.0 - mr_r2) * (n - 1) / (n - 7 - 1)
loglog_adj_r2 = 1.0 - (1.0 - loglog_r2) * (n - 1) / (n - 1 - 1)

with open("outputs/part1_results.txt", "w") as f:
    f.write(f"n = {n}\n")
    f.write(f"mean_value = {mean_val:.4f}\n")
    f.write(f"sd_value = {sd_val:.4f}\n")
    f.write(f"se_value = {se_val:.4f}\n")
    f.write(f"ci95_low = {ci_low:.4f}\n")
    f.write(f"ci95_high = {ci_high:.4f}\n")
    f.write(f"z_stat = {z_stat:.4f}\n")
    f.write(f"reject_h0 = {reject_h0}\n")
    f.write(f"n_required = {n_req}\n")
    f.write(f"reg_slope = {reg_slope:.4f}\n")
    f.write(f"reg_intercept = {reg_intercept:.4f}\n")
    f.write(f"tss = {tss:.4f}\n")
    f.write(f"ess = {ess:.4f}\n")
    f.write(f"rss = {rss:.4f}\n")
    f.write(f"tss_equals_ess_plus_rss = {tss_eq}\n")
    f.write(f"r2 = {r2:.4f}\n")
    f.write(f"rmse = {rmse:.4f}\n")
    f.write(f"mae = {mae:.4f}\n")
    f.write(f"gd_J0 = {gd_costs[0]:.4f}\n")
    f.write(f"gd_J1 = {gd_costs[1]:.4f}\n")
    f.write(f"gd_J10 = {gd_costs[10]:.4f}\n")
    f.write(f"gd_J100 = {gd_costs[100]:.4f}\n")
    f.write(f"gd_J500 = {gd_costs[500]:.4f}\n")
    f.write(f"gd_slope = {gd_slope:.4f}\n")
    f.write(f"gd_intercept = {gd_intercept:.4f}\n")
    f.write(f"gd_slope_gap = {gd_slope_gap:.4f}\n")
    f.write(f"resid_sd_lower_raw = {resid_sd_lower_raw:.4f}\n")
    f.write(f"resid_sd_upper_raw = {resid_sd_upper_raw:.4f}\n")
    f.write(f"resid_sd_ratio_raw = {resid_sd_ratio_raw:.4f}\n")
    f.write(f"loglog_intercept = {loglog_intercept:.4f}\n")
    f.write(f"loglog_slope = {loglog_slope:.4f}\n")
    f.write(f"loglog_r2 = {loglog_r2:.4f}\n")
    f.write(f"loglog_rmse = {loglog_rmse:.4f}\n")
    f.write(f"resid_sd_lower_log = {resid_sd_lower_log:.4f}\n")
    f.write(f"resid_sd_upper_log = {resid_sd_upper_log:.4f}\n")
    f.write(f"resid_sd_ratio_log = {resid_sd_ratio_log:.4f}\n")
    for i in range(8):
        f.write(f"mr_beta_{i} = {beta[i]:.4f}\n")
    f.write(f"mr_r2 = {mr_r2:.4f}\n")
    f.write(f"mr_adj_r2 = {mr_adj_r2:.4f}\n")
    f.write(f"loglog_adj_r2 = {loglog_adj_r2:.4f}\n")