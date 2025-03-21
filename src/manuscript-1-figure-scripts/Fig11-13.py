#!/usr/bin/env python3
"""
umap_dbscan_with_archive.py

This script:
  1. Loads trajectories (each a T×5 array: [x, z, heading_cos, mirrored_angle, yaw_rad]).
  2. Computes an extended 27-dimensional feature set.
  3. Scales the features, projects them with UMAP, and clusters with DBSCAN.
  4. Produces three figures (saved as PNG and PDF at 400 dpi):
       - Fig11: Average Trajectories Per Cluster
       - Fig12: 2×2 Clusters
       - Fig13: Confusion Matrix (using at most 2 sessions per player).
  5. Archives the loaded trajectories and assigned clusters if enabled.

All warnings are suppressed.
"""

import os
import pickle
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import jensenshannon
from scipy.stats import lognorm

# Suppress all warnings.
warnings.filterwarnings("ignore")

##############################
# GLOBAL CONFIGURATION
##############################
DATA_PATH = "data/structures/processed_nether_knights.pkl"

# For resampling trajectories.
RESAMPLE_LEN = 120

# UMAP and DBSCAN parameters.
N_COMPONENTS = 2
N_NEIGHBORS = 5
MIN_DIST = 0.005
EPS = 1.0
MIN_SAMPLES = 30

# Yaw shift for display.
YAW_SHIFT = np.pi / 2

# Define a simple colormap for clusters.
GLOBAL_CLUSTER_COLORS = {
    -1: "gray",   # noise
    0: "blue",
    1: "red",
    2: "green",
    3: "orange"
}
def get_cluster_color(cid):
    return GLOBAL_CLUSTER_COLORS.get(cid, "black")

##############################
# TOP-LEVEL PLOTTING PARAMETERS
##############################
N_MAJOR_TICKS = 5

# --- Fig11: Average Trajectories ---
AVG_TRAJ_FIG_SIZE         = (8, 6)
AVG_TRAJ_TITLE            = "Average Trajectories Per Cluster"
AVG_TRAJ_X_LABEL          = "X"
AVG_TRAJ_Y_LABEL          = "Z"
AVG_TRAJ_LINE_WIDTH       = 3
AVG_TRAJ_FILL_ALPHA       = 0.2
AVG_TRAJ_ARROW_EVERY      = 3    # every 3 ticks.
AVG_TRAJ_ARROW_LENGTH     = 0.3
AVG_TRAJ_ARROW_HEAD_WIDTH = 0.1
AVG_TRAJ_ARROW_HEAD_LENGTH= 0.125
BOX_SIZE = 0.717

# Hardcoded bounds for Fig11.
FIG11_XLIM = (-1.5, 3)
FIG11_YLIM = (0, 7)

# --- UMAP Plot Settings ---
UMAP_TITLE = "UMAP Projections of Trajectories Clustered Via DBSCAN"
UMAP_FIG_SIZE = (8, 6)
UMAP_ALPHA = 0.2
UMAP_POINT_SIZE = 40
UMAP_TICK_COUNT = 5

# --- Fig13: Confusion Matrix ---
CONF_MATRIX_FIG_SIZE   = (6, 5)
CONF_MATRIX_TITLE      = "Confusion Matrix (Nearest-Average)"
CONF_MATRIX_X_LABEL    = "Predicted PID"
CONF_MATRIX_Y_LABEL    = "True PID"

##############################
# MANUSCRIPT FIGURE OUTPUT DIRECTORIES
##############################
MANUSCRIPT_DIR = "manuscript-1-figures"
FIG11_20_21_22_DIR = os.path.join(MANUSCRIPT_DIR, "Fig11-13")
FIG11_DIR = os.path.join(FIG11_20_21_22_DIR, "Fig11")
FIG12_DIR = os.path.join(FIG11_20_21_22_DIR, "Fig12")
FIG13_DIR = os.path.join(FIG11_20_21_22_DIR, "Fig13")
os.makedirs(FIG11_DIR, exist_ok=True)
os.makedirs(FIG12_DIR, exist_ok=True)
os.makedirs(FIG13_DIR, exist_ok=True)

# --- ARCHIVING ---
ARCHIVE_DATA = True
ARCHIVE_PATH = os.path.join(FIG11_20_21_22_DIR, "archived_data.pkl")

##############################
# 1) Load Data
##############################
def load_trajectories(data_path):
    with open(data_path, "rb") as f:
        final_data = pickle.load(f)
    all_trajs = []
    all_metadata = []
    for traj in final_data:
        x_arr    = traj.get("mirrored_x")
        z_arr    = traj.get("mirrored_z")
        cos_arr  = traj.get("mirrored_heading")
        ang_arr  = traj.get("mirrored_angle")
        yaw_deg  = traj.get("mirrored_yaw")
        pid      = traj.get("pid")
        trial_id = traj.get("trial_id")
        if any(v is None for v in [x_arr, z_arr, cos_arr, ang_arr, yaw_deg]):
            continue
        min_len = min(len(x_arr), len(z_arr), len(cos_arr), len(ang_arr), len(yaw_deg))
        if min_len < 2:
            continue
        x_arr   = x_arr[:min_len]
        z_arr   = z_arr[:min_len]
        cos_arr = cos_arr[:min_len]
        ang_arr = ang_arr[:min_len]
        yaw_deg = yaw_deg[:min_len]
        yaw_rad = np.deg2rad(yaw_deg)
        data_5col = np.column_stack([x_arr, z_arr, cos_arr, ang_arr, yaw_rad]).astype(np.float32)
        if np.isnan(data_5col).any():
            continue
        all_trajs.append(data_5col)
        all_metadata.append({"pid": pid, "trial_id": trial_id})
    print(f"[LOAD] Found {len(all_trajs)} valid trajectories from {data_path}.")
    return all_trajs, all_metadata

##############################
# 2) Compute Extended Features (27D)
##############################
def compute_extended_features(all_trajs):
    feats_list = []
    for traj in all_trajs:
        x_vals   = traj[:, 0]
        z_vals   = traj[:, 1]
        cos_vals = traj[:, 2]
        ang_vals = traj[:, 3]
        T = len(traj)
        if T < 2:
            continue
        i_min_x   = np.argmin(x_vals)
        min_x     = x_vals[i_min_x]
        i_min_cos = np.argmin(cos_vals)
        min_cos   = cos_vals[i_min_cos]
        t_min_x   = i_min_x / (T - 1)
        z_min_x   = z_vals[i_min_x]
        t_min_cos = i_min_cos / (T - 1)
        z_min_cos = z_vals[i_min_cos]
        x_min_cos = x_vals[i_min_cos]
        i_min_ang = np.argmin(ang_vals)
        min_ang   = ang_vals[i_min_ang]
        num_neg   = np.sum(x_vals < 0)
        pct_x_negative = num_neg / float(T)
        dx = np.diff(x_vals)
        dz = np.diff(z_vals)
        total_distance = np.sum(np.sqrt(dx**2 + dz**2))
        x_final = x_vals[-1]
        z_final = z_vals[-1]
        i_max_cos = np.argmax(cos_vals)
        max_cos   = cos_vals[i_max_cos]
        t_max_cos = i_max_cos / (T - 1)
        x_max_cos = x_vals[i_max_cos]
        z_max_cos = z_vals[i_max_cos]
        i_max_ang = np.argmax(ang_vals)
        max_ang   = ang_vals[i_max_ang]
        t_max_ang = i_max_ang / (T - 1)
        x_max_ang = x_vals[i_max_ang]
        z_max_ang = z_vals[i_max_ang]
        mean_cos  = np.mean(cos_vals)
        std_cos   = np.std(cos_vals)
        mean_ang  = np.mean(ang_vals)
        std_ang   = np.std(ang_vals)
        net_x     = x_final - x_vals[0]
        net_z     = z_final - z_vals[0]
        pct_heading_positive = np.sum(cos_vals > 0) / float(T)
        feats = np.array([
            min_x, min_cos, t_min_x, z_min_x, t_min_cos, z_min_cos, x_min_cos,
            min_ang, pct_x_negative, total_distance, x_final, z_final,
            max_cos, t_max_cos, x_max_cos, z_max_cos, max_ang, t_max_ang,
            x_max_ang, z_max_ang, mean_cos, std_cos, mean_ang, std_ang,
            net_x, net_z, pct_heading_positive
        ], dtype=np.float32)
        feats_list.append(feats)
    X_features = np.vstack(feats_list) if feats_list else np.array([])
    print(f"[FEATURES] Extended features shape: {X_features.shape}")
    return X_features

##############################
# 3) UMAP Embedding and DBSCAN Clustering
##############################
def umap_embedding(X, n_neighbors, min_dist, n_components):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        n_components=n_components, random_state=42)
    emb = reducer.fit_transform(X)
    print(f"[UMAP] Embedding shape: {emb.shape}")
    return emb

def dbscan_cluster(emb, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm="ball_tree")
    labels = db.fit_predict(emb)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"[DBSCAN] Found {num_clusters} clusters (noise included as -1).")
    return labels

##############################
# 4) Plotting Functions
##############################
def resample_traj(traj, new_len=RESAMPLE_LEN):
    T = len(traj)
    if T < 2:
        return None
    old_time = np.linspace(0, 1, T)
    new_time = np.linspace(0, 1, new_len)
    out = np.zeros((new_len, 5), dtype=np.float32)
    for c in range(5):
        out[:, c] = np.interp(new_time, old_time, traj[:, c])
    return out

def plot_average_trajectories(all_trajs, labels, yaw_shift, out_dir):
    plt.figure(figsize=AVG_TRAJ_FIG_SIZE)
    ax = plt.gca()
    handles = []
    for cid in sorted(np.unique(labels)):
        idxs = np.where(labels == cid)[0]
        if cid == -1 or len(idxs) <= 100:
            continue
        color = get_cluster_color(cid)
        resampled_list = []
        for i in idxs:
            resampled = resample_traj(all_trajs[i], RESAMPLE_LEN)
            if resampled is not None:
                resampled_list.append(resampled)
        if not resampled_list:
            continue
        stack_ = np.stack(resampled_list, axis=0)
        x_all = stack_[:, :, 0]
        z_all = stack_[:, :, 1]
        yaw_all = stack_[:, :, 4]
        x_avg = np.mean(x_all, axis=0)
        z_avg = np.mean(z_all, axis=0)
        x_25 = np.percentile(x_all, 25, axis=0)
        x_75 = np.percentile(x_all, 75, axis=0)
        polygon_x = np.concatenate([x_25, x_75[::-1]])
        polygon_z = np.concatenate([z_avg, z_avg[::-1]])
        ax.fill(polygon_x, polygon_z, color=color, alpha=AVG_TRAJ_FILL_ALPHA)
        line, = ax.plot(x_avg, z_avg, color=color, linewidth=AVG_TRAJ_LINE_WIDTH, 
                        label=f"Type {['DG', 'IN', 'IG', 'DN'][cid]} (N={len(idxs)})")
        handles.append(line)
        # Yaw arrow averaging
        yaw_avg = np.zeros(RESAMPLE_LEN, dtype=np.float32)
        for t in range(RESAMPLE_LEN):
            yaws_t = yaw_all[:, t]
            s_ = np.sin(yaws_t)
            c_ = np.cos(yaws_t)
            yaw_avg[t] = np.arctan2(np.mean(s_), np.mean(c_)) + yaw_shift
        for t in range(0, RESAMPLE_LEN, AVG_TRAJ_ARROW_EVERY):
            dx = -AVG_TRAJ_ARROW_LENGTH * np.cos(yaw_avg[t])
            dz = AVG_TRAJ_ARROW_LENGTH * np.sin(yaw_avg[t])
            ax.arrow(x_avg[t], z_avg[t], dx, dz,
                     head_width=AVG_TRAJ_ARROW_HEAD_WIDTH,
                     head_length=AVG_TRAJ_ARROW_HEAD_LENGTH,
                     fc=color, ec=color, lw=1, length_includes_head=True)
    ax.set_title(AVG_TRAJ_TITLE)
    ax.set_xlabel(AVG_TRAJ_X_LABEL)
    ax.set_ylabel(AVG_TRAJ_Y_LABEL)
    ax.set_xlim(FIG11_XLIM)
    ax.set_ylim(FIG11_YLIM)
    ax.set_aspect('equal', adjustable='box')
    square_side = BOX_SIZE
    square_bl = (2.5 - square_side/2, 6 - square_side/2)
    ax.add_patch(patches.Rectangle(square_bl, square_side, square_side,
                                   facecolor='gray', edgecolor='none'))
    ax.legend(handles=handles, loc='upper left')
    folder_name = os.path.basename(os.path.normpath(out_dir))
    base_path = os.path.join(out_dir, folder_name)
    plt.savefig(base_path + ".png", dpi=400, bbox_inches='tight')
    plt.savefig(base_path + ".pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Average trajectories saved as {base_path}.png and {base_path}.pdf")

def plot_umap_clusters(embedding, labels, out_dir):
    plt.figure(figsize=UMAP_FIG_SIZE)
    valid_labels = [0, 1, 2, 3]
    for lab in valid_labels:
        idxs = np.where(labels == lab)[0]
        if len(idxs) == 0:
            continue
        color = get_cluster_color(lab)
        plt.scatter(embedding[idxs, 0], embedding[idxs, 1],
                    c=color, label=f"Type {['DG', 'IN', 'IG', 'DN'][lab]}",
                    alpha=UMAP_ALPHA, s=UMAP_POINT_SIZE)
    valid_idxs = np.where(np.isin(labels, valid_labels))[0]
    if len(valid_idxs) > 0:
        x_min_raw = np.min(embedding[valid_idxs, 0])
        x_max_raw = np.max(embedding[valid_idxs, 0])
        new_x_min = int(np.floor(x_min_raw))
        new_x_max = int(np.ceil(x_max_raw))
        plt.xlim(new_x_min, new_x_max)
        xticks = np.linspace(new_x_min, new_x_max, num=UMAP_TICK_COUNT)
        plt.xticks(xticks)
        y_min_raw = np.min(embedding[valid_idxs, 1])
        y_max_raw = np.max(embedding[valid_idxs, 1])
        new_y_min = int(np.floor(y_min_raw))
        new_y_max = int(np.ceil(y_max_raw))
        plt.ylim(new_y_min, new_y_max)
        yticks = np.linspace(new_y_min, new_y_max, num=UMAP_TICK_COUNT)
        plt.yticks(yticks)
    plt.title(UMAP_TITLE)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.legend(loc="best", markerscale=1)
    plt.tight_layout()
    folder_name = os.path.basename(os.path.normpath(out_dir))
    base_path = os.path.join(out_dir, folder_name)
    plt.savefig(base_path + ".png", dpi=400, bbox_inches='tight')
    plt.savefig(base_path + ".pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] UMAP clusters (0–3 only) saved as {base_path}.png and {base_path}.pdf")

##############################
# 5) Equal-Weight Identifiability & Bootstrapping
##############################
def compute_equal_weight_identifiability_metric(filtered_dist):
    """
    Computes the per-participant identifiability score as the average vote across
    that participant's sessions. Each session's vote is:
      - 1.0 if the session's profile is closest to its true participant's average,
      - 0.5 if second closest,
      - 0.25 if third closest,
      - 0 otherwise.
    The final score is the mean of all participants' scores, converted to %.
    """
    from collections import defaultdict
    pid_to_sessions = defaultdict(list)
    for (pid, sess), vec in filtered_dist.items():
        pid_to_sessions[pid].append(vec)
    valid_pids = [pid for pid, sess_list in pid_to_sessions.items() if len(sess_list) >= 2]
    if not valid_pids:
        return None

    # Compute each participant's average distribution.
    pid_avg = {pid: np.mean(np.vstack(pid_to_sessions[pid]), axis=0) for pid in valid_pids}
    per_participant_scores = []
    for pid in valid_pids:
        votes = []
        for v in pid_to_sessions[pid]:
            distances = {candidate: jenshenson(v, pid_avg[candidate]) for candidate in valid_pids}  # We'll define jenshenson
            sorted_candidates = sorted(distances, key=distances.get)
            if sorted_candidates[0] == pid:
                vote = 1.0
            elif len(sorted_candidates) > 1 and sorted_candidates[1] == pid:
                vote = 0.5
            elif len(sorted_candidates) > 2 and sorted_candidates[2] == pid:
                vote = 0.25
            else:
                vote = 0.0
            votes.append(vote)
        per_participant_scores.append(np.mean(votes))
    return np.mean(per_participant_scores) * 100.0

def jenshenson(v1, v2):
    """Convenience function for Jensen–Shannon distance."""
    return jensenshannon(v1, v2)

def compute_equal_weight_per_participant_scores(filtered_dist):
    """
    Returns a dict mapping each participant (≥2 sessions) to their per-participant score.
    """
    from collections import defaultdict
    pid_to_sessions = defaultdict(list)
    for (pid, sess), vec in filtered_dist.items():
        pid_to_sessions[pid].append(vec)
    valid_pids = [pid for pid, s in pid_to_sessions.items() if len(s) >= 2]
    if not valid_pids:
        return {}
    pid_avg = {pid: np.mean(np.vstack(pid_to_sessions[pid]), axis=0) for pid in valid_pids}
    scores = {}
    for pid in valid_pids:
        votes = []
        for v in pid_to_sessions[pid]:
            distances = {candidate: jenshenson(v, pid_avg[candidate]) for candidate in valid_pids}
            sorted_candidates = sorted(distances, key=distances.get)
            if sorted_candidates[0] == pid:
                vote = 1.0
            elif len(sorted_candidates) > 1 and sorted_candidates[1] == pid:
                vote = 0.5
            elif len(sorted_candidates) > 2 and sorted_candidates[2] == pid:
                vote = 0.25
            else:
                vote = 0.0
            votes.append(vote)
        scores[pid] = np.mean(votes)
    return scores

def bootstrap_equal_weight_metric(filtered_dist, n_bootstrap=1000):
    """
    Bootstraps the equal-weight identifiability metric by resampling participants
    with replacement. Returns an array of bootstrapped overall metrics (in %).
    """
    scores = compute_equal_weight_per_participant_scores(filtered_dist)
    participants = list(scores.keys())
    if not participants:
        return None
    boot_metrics = []
    for _ in range(n_bootstrap):
        sampled = np.random.choice(participants, size=len(participants), replace=True)
        boot_metric = np.mean([scores[pid] for pid in sampled]) * 100.0
        boot_metrics.append(boot_metric)
    return np.array(boot_metrics)

def equal_weight_permutation_test(filtered_dist, num_permutations=1000):
    """
    Shuffles participant labels among sessions and computes the equal-weight
    identifiability metric for each permutation. Returns observed, distribution, p-value.
    """
    observed_metric = compute_equal_weight_identifiability_metric(filtered_dist)
    metrics = []
    keys = list(filtered_dist.keys())
    true_pids = [k[0] for k in keys]
    for _ in range(num_permutations):
        permuted_pids = np.random.permutation(true_pids)
        permuted_dict = {}
        for new_pid, key in zip(permuted_pids, keys):
            permuted_dict[(new_pid, key[1])] = filtered_dist[key]
        metric = compute_equal_weight_identifiability_metric(permuted_dict)
        if metric is not None:
            metrics.append(metric)
    metrics = np.array(metrics)
    p_value = np.mean(metrics >= observed_metric)
    return observed_metric, metrics, p_value

##############################
# 6) Confusion Matrix + Plotting
##############################
def plot_confusion_matrix_identifiability(dist_dict, out_dir):
    """
    Fig13: Plots a "weighted confusion matrix" in terms of raw distribution matches,
    but does NOT do any extra permutations. It's just for a quick visual of the
    session-level data.
    """
    from collections import defaultdict
    import seaborn as sns

    sessions_by_pid = defaultdict(list)
    for (pid, sess), vec in dist_dict.items():
        sessions_by_pid[pid].append(((pid, sess), vec))
    
    filtered_dist = {}
    for pid, items in sessions_by_pid.items():
        items = sorted(items, key=lambda x: x[0][1])
        for key, vec in items:
            filtered_dist[key] = vec
    
    pid_to_sessions = defaultdict(list)
    for (pid, sess), vec in filtered_dist.items():
        pid_to_sessions[pid].append(vec)
    filtered_pids = {pid for pid, lst in pid_to_sessions.items() if len(lst) > 1}
    if not filtered_pids:
        print("[Confusion Matrix] No players with multiple sessions available.")
        return
    
    # Compute each player's average distribution for display.
    pid_averages = {}
    for pid in filtered_pids:
        lst = pid_to_sessions[pid]
        pid_averages[pid] = np.mean(np.vstack(lst), axis=0)

    unique_pids = sorted(list(pid_averages.keys()))
    pid_index = {p: i for i, p in enumerate(unique_pids)}
    num_players = len(unique_pids)
    
    # Build confusion matrix based on nearest average (simple approach).
    votes = np.zeros((num_players, num_players), dtype=float)
    total_votes = np.zeros(num_players, dtype=float)
    
    for (true_pid, sess), vec in filtered_dist.items():
        if true_pid not in filtered_pids:
            continue
        i_true = pid_index[true_pid]
        distances = {candidate: jenshenson(vec, pid_averages[candidate])
                     for candidate in unique_pids}
        sorted_candidates = sorted(distances, key=distances.get)
        predicted = sorted_candidates[0]
        i_pred = pid_index[predicted]
        votes[i_true, i_pred] += 1.0
        vote_total = 1.0
        # Partial votes for 2nd/3rd place are optional; remove if desired
        if predicted != true_pid:
            if len(sorted_candidates) > 1 and sorted_candidates[1] == true_pid:
                votes[i_true, i_true] += 0.5
                vote_total += 0.5
            elif len(sorted_candidates) > 2 and sorted_candidates[2] == true_pid:
                votes[i_true, i_true] += 0.25
                vote_total += 0.25
        total_votes[i_true] += vote_total
    
    conf_mat_norm = np.zeros_like(votes)
    for i in range(num_players):
        if total_votes[i] > 0:
            conf_mat_norm[i, :] = (votes[i, :] / total_votes[i]) * 100.0

    conf_mat_norm = np.floor(conf_mat_norm)
    annot = np.where(conf_mat_norm == 0, "", conf_mat_norm.astype(int).astype(str))
    row_labels = [p[-3:] for p in unique_pids]
    
    plt.figure(figsize=CONF_MATRIX_FIG_SIZE)
    sns.heatmap(conf_mat_norm, annot=annot, fmt="", cmap="Blues",
                xticklabels=row_labels, yticklabels=row_labels,
                annot_kws={"fontsize": 8})
    plt.title("Confusion Matrix (Nearest-Average)")
    plt.xlabel(CONF_MATRIX_X_LABEL)
    plt.ylabel(CONF_MATRIX_Y_LABEL)
    folder_name = os.path.basename(os.path.normpath(out_dir))
    base_path = os.path.join(out_dir, folder_name)
    plt.tight_layout()
    plt.savefig(base_path + ".png", dpi=400, bbox_inches='tight')
    plt.savefig(base_path + ".pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Confusion matrix saved as {base_path}.png and {base_path}.pdf")

##############################
# 7) Plotting the Permutation + Bootstrap
##############################
def plot_permutation_with_bootstrap(observed_metric, permuted_metrics, p_value,
                                    boot_ci, out_dir, test_name="equal_weight"):
    """
    Plots the permutation distribution, marks the observed metric, p-value,
    and overlays vertical lines for the 95% CI from bootstrapping.
    """
    plt.figure(figsize=(6,4))
    plt.hist(permuted_metrics, bins=30, alpha=0.7, label='Permutation distribution')
    plt.axvline(observed_metric, color='red', linestyle='dashed', linewidth=2,
                label=f'Observed (p={p_value:.3f})')
    
    # Overlay 95% CI from bootstrapping in green lines
    if boot_ci is not None:
        ci_lower, ci_upper = boot_ci
        plt.axvline(ci_lower, color='green', linestyle='dotted', linewidth=2,
                    label='95% CI lower')
        plt.axvline(ci_upper, color='green', linestyle='dotted', linewidth=2,
                    label='95% CI upper')
    
    plt.xlabel('Mean Diagonal Percentage')
    plt.ylabel('Frequency')
    plt.title(f'Permutation Test (Equal-Weight) with Bootstrap CI')
    plt.legend()
    base_path = os.path.join(out_dir, f"Fig13_permutation_{test_name}")
    plt.savefig(base_path + ".png", dpi=400, bbox_inches='tight')
    plt.savefig(base_path + ".pdf", dpi=400, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Permutation + Bootstrap CI saved as {base_path}.png and {base_path}.pdf")

def build_breakdown_dict_for_reliability(all_metadata, labels):
    breakdown = {}
    for i, meta in enumerate(all_metadata):
        pid = meta["pid"]
        trial_id = meta["trial_id"]
        parts = trial_id.split("_")
        session = parts[2] if len(parts) >= 3 else "UNKNOWN_SESSION"
        cid = labels[i]
        key = (pid, session)
        if key not in breakdown:
            breakdown[key] = {}
        breakdown[key][cid] = breakdown[key].get(cid, 0) + 1
    unique_clusters = sorted(list(set(labels)))
    return breakdown, unique_clusters

def get_cluster_distribution_vectors(breakdown_dict, cluster_labels):
    out = {}
    for (pid, sess), cdict in breakdown_dict.items():
        total = sum(cdict.values())
        vec = [cdict.get(cl, 0) / total if total > 0 else 0.0 for cl in cluster_labels]
        out[(pid, sess)] = np.array(vec, dtype=float)
    return out

def get_filtered_distribution(dist_dict):
    """
    Returns the filtered dictionary of sessions:
      - For each player, take at most 2 sessions (sorted by session ID)
      - Only players with >1 session are kept.
    """
    from collections import defaultdict
    sessions_by_pid = defaultdict(list)
    for (pid, sess), vec in dist_dict.items():
        sessions_by_pid[pid].append(((pid, sess), vec))
    filtered_dist = {}
    for pid, items in sessions_by_pid.items():
        items = sorted(items, key=lambda x: x[0][1])
        for key, vec in items:
            filtered_dist[key] = vec
    pid_to_sessions = defaultdict(list)
    for (pid, sess), vec in filtered_dist.items():
        pid_to_sessions[pid].append(vec)
    filtered_pids = {pid for pid, lst in pid_to_sessions.items() if len(lst) > 1}
    if not filtered_pids:
        return None
    return filtered_dist

##############################
# MAIN
##############################
def main():
    # 1) Load data
    all_trajs, all_metadata = load_trajectories(DATA_PATH)
    if len(all_trajs) == 0:
        print("No data => exit.")
        return

    # 2) Compute extended features
    X_extended = compute_extended_features(all_trajs)
    if X_extended.size == 0:
        print("No valid features => exit.")
        return

    # 3) UMAP + DBSCAN
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_extended)
    emb_2d = umap_embedding(X_scaled, n_neighbors=N_NEIGHBORS,
                            min_dist=MIN_DIST, n_components=N_COMPONENTS)
    labels = dbscan_cluster(emb_2d, eps=EPS, min_samples=MIN_SAMPLES)
    cluster_color_map = {cid: get_cluster_color(cid) for cid in np.unique(labels)}

    # 4) Plots
    plot_average_trajectories(all_trajs, labels, yaw_shift=YAW_SHIFT, out_dir=FIG11_DIR)
    plot_umap_clusters(emb_2d, labels, out_dir=FIG12_DIR)
    
    # 5) Build distribution dict for reliability & plot confusion matrix
    breakdown_dict, unique_clusters = build_breakdown_dict_for_reliability(all_metadata, labels)
    dist_dict = get_cluster_distribution_vectors(breakdown_dict, unique_clusters)
    plot_confusion_matrix_identifiability(dist_dict, out_dir=FIG13_DIR)

    # 6) Equal-Weight Permutation Test + Bootstrapping
    filtered_dist = get_filtered_distribution(dist_dict)
    if filtered_dist is not None:
        obs_eq, perm_eq, p_eq = equal_weight_permutation_test(filtered_dist, num_permutations=100000)
        print(f"[EQUAL-WEIGHT PERMUTATION TEST] Observed mean diagonal: {obs_eq:.2f}%, p-value: {p_eq:.3f}")
        cohens_d = (obs_eq - np.mean(perm_eq)) / np.std(perm_eq)
        print(f"[EQUAL-WEIGHT EFFECT SIZE] Cohen's d: {cohens_d:.2f}")

        # Run bootstrapping
        boot_metrics = bootstrap_equal_weight_metric(filtered_dist, n_bootstrap=1)
        if boot_metrics is not None:
            ci_lower = np.percentile(boot_metrics, 2.5)
            ci_upper = np.percentile(boot_metrics, 97.5)
            print(f"[BOOTSTRAP] 95% CI for equal-weight metric: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
            ci_tuple = (ci_lower, ci_upper)
        else:
            ci_tuple = None
        
        # Plot permutation distribution with the observed metric & bootstrap CI
        plot_permutation_with_bootstrap(obs_eq, perm_eq, p_eq, ci_tuple, FIG13_DIR, test_name="equal_weight")
    else:
        print("Not enough players with multiple sessions for equal-weight permutation test.")

    # 7) Archive if desired
    if ARCHIVE_DATA:
        archive = {"trajectories": all_trajs, "labels": labels}
        with open(ARCHIVE_PATH, "wb") as f:
            pickle.dump(archive, f)
        print(f"[ARCHIVE] Archived data saved to {ARCHIVE_PATH}")

    print("Done. Check the manuscript-1-figures directories for Fig11–23.")

if __name__ == "__main__":
    main()
