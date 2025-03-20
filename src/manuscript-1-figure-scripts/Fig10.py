#!/usr/bin/env python3

import sys
import warnings
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
import statsmodels.api as sm
from pathlib import Path

# For Bayes factor on Pearson r
import pingouin as pg

###########################
# Adjust project_root as needed
###########################
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data_structures.game_data import GameDataManager

##############################################################################
# GLOBAL PARAMETERS
##############################################################################
FIG_SIZE = 5.0
MARKER_SIZE = 150.0
MARKER_ALPHA = 0.5
REG_FILL_ALPHA = 0.2
MARKER_COLOR = "steelblue"
REG_LINE_WIDTH = 3.5
REG_COLOR = "darkorange"

AXIS_LABEL_FONT_SIZE = 14
TICK_LABEL_FONT_SIZE = 16
SUPER_TITLE_FONT_SIZE = 16

N_MAJOR_TICKS = 5
TICK_LENGTH = 6
TICK_WIDTH = 1
SPINE_LINEWIDTH = 1

# For axis buffering and tick rounding
AXIS_BUFFER_RATIO = 0.05
DECIMALS_FOR_TICKS = 1

# Fix a random seed for jitter
random.seed(42)

##############################################################################
# LOGISTIC FUNCTION
##############################################################################
def logistic(x, x0, spread):
    """
    Logistic function parameterized by psi_theta (x0) and spread.
    Returns success rate in percentage terms.
    """
    k = 2 * np.log(3) / spread
    return 100 / (1 + np.exp(k * (x - x0)))


def _style_spines_and_ticks(ax):
    """
    Helper to match sample script's black spines & outward ticks.
    """
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(SPINE_LINEWIDTH)
        ax.spines[spine].set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        colors="black",
        labelsize=TICK_LABEL_FONT_SIZE
    )
##############################################################################
# SESSION MAPPING
##############################################################################
def build_session_mapping(manager: GameDataManager):
    """
    Returns a nested dict: session_map[pid][session_id] = session_number,
    sorted by ascending timestamp across all sessions.
    """
    session_map = {}
    for pid, player_data in manager.players.items():
        session_list = []
        for game_data in player_data.games.values():
            for sid, sess in game_data.sessions.items():
                session_list.append((sess.timestamp, sid))
        session_list.sort(key=lambda x: x[0])  # sort by timestamp
        session_map[pid] = {}
        for i, (ts, sid) in enumerate(session_list):
            session_map[pid][sid] = i + 1
    return session_map

##############################################################################
# FIT FUNCTIONS FOR BUILD MASTER
##############################################################################
def fit_buildmaster_session(session_id, diff_map):
    # same as your existing code, not repeated here for brevity
    if len(diff_map) < 2:
        return None

    x_data = np.array(sorted(diff_map.keys()), dtype=float)
    y_data = np.array([
        (diff_map[d]["successes"] / diff_map[d]["trials"]) * 100.0
        for d in x_data
    ])

    # Attempt curve_fit ...
    try:
        popt, _ = curve_fit(
            logistic, x_data, y_data,
            p0=[5.0, 1.0],
            bounds=([0, 0.5], [11, 10]),
            maxfev=5000
        )
    except RuntimeError:
        # fallback attempts ...
        try:
            init_psi_theta = np.interp(50, np.flip(y_data), np.flip(x_data))
            init_psi_theta = min(init_psi_theta, 10)
            popt, _ = curve_fit(
                logistic, x_data, y_data,
                p0=[init_psi_theta, 2.0],
                bounds=([0, 0.1], [11, 10]),
                maxfev=5000
            )
        except RuntimeError:
            init_psi_theta = np.interp(50, np.flip(y_data), np.flip(x_data))
            init_psi_theta = min(init_psi_theta, 11)
            rmse = np.sqrt(np.mean((y_data - init_psi_theta)**2))
            x0 = min(init_psi_theta, 10.5)
            return {
                "psi_theta": x0,
                "spread": 2.0,
                "rmse": rmse,
                "session_id": session_id,
            }

    x0, spread = popt
    x0 = min(x0, 10.5)
    y_fit = logistic(x_data, x0, spread)
    rmse = np.sqrt(np.mean((y_data - y_fit)**2))

    return {
        "psi_theta": x0,
        "spread": spread,
        "rmse": rmse,
        "session_id": session_id,
    }

def get_buildmaster_fits_by_session(player_data, game_mode):
    # gather session-level fits
    bm_game = player_data.get_game("BUILD_MASTER")
    if not bm_game:
        return []
    out = []
    for sid, sess in bm_game.sessions.items():
        # relevant trials for that mode
        relevant = [t for t in sess.trials
                    if t.additional_info.get("game_mode") == game_mode]
        diff_map = {}
        for att in relevant:
            d = att.difficulty
            if d is None:
                continue
            if d not in diff_map:
                diff_map[d] = {"successes": 0, "trials": 0}
            diff_map[d]["trials"] += 1
            if att.success:
                diff_map[d]["successes"] += 1
        res = fit_buildmaster_session(sid, diff_map)
        if res:
            out.append(res)
    return out

def get_first_bm_fit(player_data, game_mode, session_map):
    # earliest BM fit by session number
    fits = get_buildmaster_fits_by_session(player_data, game_mode)
    if not fits:
        return None
    pid = player_data.pid
    best = None
    best_val = 999999
    for fobj in fits:
        s_id = fobj["session_id"]
        s_num = session_map[pid].get(s_id, 999999)
        if s_num < best_val:
            best_val = s_num
            best = fobj
    return best

##############################################################################
# FIT FUNCTIONS FOR LSWM
##############################################################################
def fit_lswm_session(session_id, diff_map):
    # same logic for LSWM with x0 capped at 7.5
    if len(diff_map) < 2:
        return None
    x_data = np.array(sorted(diff_map.keys()), dtype=float)
    y_data = np.array([
        (diff_map[d]["successes"] / diff_map[d]["trials"]) * 100.0
        for d in x_data
    ])
    try:
        popt, _ = curve_fit(
            logistic, x_data, y_data,
            p0=[3.0, 1.0],
            bounds=([0, 0.5], [15, 10]),
            maxfev=5000
        )
    except RuntimeError:
        try:
            init_psi_theta = np.interp(50, np.flip(y_data), np.flip(x_data))
            init_psi_theta = min(init_psi_theta, 14)
            popt, _ = curve_fit(
                logistic, x_data, y_data,
                p0=[init_psi_theta, 2.0],
                bounds=([0, 0.1], [15, 10]),
                maxfev=5000
            )
        except RuntimeError:
            init_psi_theta = np.interp(50, np.flip(y_data), np.flip(x_data))
            init_psi_theta = min(init_psi_theta, 15)
            rmse = np.sqrt(np.mean((y_data - init_psi_theta)**2))
            x0 = min(init_psi_theta, 7.5)
            return {
                "psi_theta": x0,
                "spread": 2.0,
                "rmse": rmse,
                "session_id": session_id,
            }
    x0, spread = popt
    x0 = min(x0, 7.5)
    y_fit = logistic(x_data, x0, spread)
    rmse = np.sqrt(np.mean((y_data - y_fit)**2))
    return {
        "psi_theta": x0,
        "spread": spread,
        "rmse": rmse,
        "session_id": session_id,
    }

def get_lswm_fits_by_session(player_data, list_number):
    lswm_game = player_data.get_game("List Sorting Working Memory")
    if not lswm_game:
        return []
    prefix = f"LSWM_{list_number}List_"
    out = []
    for sid, sess in lswm_game.sessions.items():
        relevant = []
        for t in sess.trials:
            item_id = t.additional_info.get("ItemID", "")
            if prefix in item_id:
                relevant.append(t)
        diff_map = {}
        for att in relevant:
            d = att.difficulty
            if d is None:
                continue
            if d not in diff_map:
                diff_map[d] = {"successes": 0, "trials": 0}
            diff_map[d]["trials"] += 1
            if att.success:
                diff_map[d]["successes"] += 1
        res = fit_lswm_session(sid, diff_map)
        if res:
            out.append(res)
    return out

def analyze_participant_contributions(df: pd.DataFrame) -> None:
    """
    Print a summary of participant contributions from the correlation dataframe.
    """
    # Get unique participants and their contributions
    unique_pids = df[["BM_Mode", "LSWM_Form", "BM_psi"]].groupby(["BM_Mode", "LSWM_Form"]).count()
    print("\nParticipant Contribution Summary:")
    print("-" * 40)
    
    # Print counts for each combination
    for (bm_mode, form), count in unique_pids.iterrows():
        print(f"Build Master {bm_mode} vs LSWM Form {form}: {count['BM_psi']} participants")

##############################################################################
# CORRELATION PLOT
##############################################################################
def correlate_bm_vs_lswm_first_bm(manager: GameDataManager, build_modes, output_dir: Path):
    """
    - Gathers the FIRST Build Master psi_theta for each player (across build_modes)
      vs. ALL LSWM psi_thetas (for forms 1 & 2).
    - Plots subplots (rows=len(build_modes), cols=2).
    - Uses the axis buffering method (rounding to 1 decimal place).
    - No participant labels.
    - Title: "Paired $\psi_{\theta}$ for Rainbow Random and List Sort"
    - X label: "Rainbow Random $\psi_{\theta}$"
    - Y label: "List Sort Form 1 $\psi_{\theta}$" or "List Sort Form 2 $\psi_{\theta}$"
    """
    session_map = build_session_mapping(manager)
    lswm_forms = [1, 2]

    # gather data
    rows = []
    for pid, player_data in manager.players.items():
        for bm_mode in build_modes:
            bm_fit = get_first_bm_fit(player_data, bm_mode, session_map)
            if not bm_fit:
                continue
            bm_val = bm_fit["psi_theta"]
            for lf in lswm_forms:
                all_lswm = get_lswm_fits_by_session(player_data, lf)
                if not all_lswm:
                    continue
                for lobj in all_lswm:
                    rows.append({
                        "PID": pid,
                        "BM_Mode": bm_mode,
                        "BM_psi": bm_val,
                        "LSWM_Form": lf,
                        "LSWM_psi": lobj["psi_theta"],
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data to plot.")
        return

    # Print contribution summary
    analyze_participant_contributions(df)
    
    # Save detailed contribution data
    contribution_path = output_dir / "participant_contributions.csv"
    df[["PID", "BM_Mode", "LSWM_Form"]].drop_duplicates().to_csv(contribution_path, index=False)
    print(f"\nSaved detailed contribution data to: {contribution_path}")

    # figure layout
    n_rows = len(build_modes)
    n_cols = len(lswm_forms)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(FIG_SIZE * n_cols, FIG_SIZE * n_rows),
        squeeze=False
    )
    fig.suptitle("Paired $\\psi_{\\theta}$ for Rainbow Random and List Sort",
                 fontsize=SUPER_TITLE_FONT_SIZE, y=0.98)

    import math

    def round_down(value, decimals=1):
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def round_up(value, decimals=1):
        factor = 10 ** decimals
        return math.ceil(value * factor) / factor

    CLOSE_THRESH = 0.25
    MAX_JITTER = 0.1

    for r, bm_mode in enumerate(build_modes):
        for c, form_num in enumerate(lswm_forms):
            ax = axes[r, c]
            sub = df[(df["BM_Mode"] == bm_mode) & (df["LSWM_Form"] == form_num)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                _style_spines_and_ticks(ax)
                ax.set_xlabel("Rainbow Random $\\psi_{\\theta}$", fontsize=AXIS_LABEL_FONT_SIZE)
                ax.set_ylabel(f"List Sort Form {form_num} $\\psi_{{\\theta}}$", fontsize=AXIS_LABEL_FONT_SIZE)
                continue

            x_data = sub["BM_psi"].values
            y_data = sub["LSWM_psi"].values

            # Jitter overlapping points
            x_plot = x_data.copy()
            y_plot = y_data.copy()
            used = np.zeros(len(x_data), dtype=bool)
            for i in range(len(x_data)):
                if used[i]:
                    continue
                cluster = [i]
                for j in range(i+1, len(x_data)):
                    if used[j]:
                        continue
                    dx = abs(x_data[i] - x_data[j])
                    dy = abs(y_data[i] - y_data[j])
                    if dx < CLOSE_THRESH and dy < CLOSE_THRESH:
                        cluster.append(j)
                if len(cluster) > 1:
                    for cc in cluster:
                        used[cc] = True
                        rx = random.uniform(-MAX_JITTER, MAX_JITTER)
                        ry = random.uniform(-MAX_JITTER, MAX_JITTER)
                        x_plot[cc] += rx
                        y_plot[cc] += ry
                else:
                    used[i] = True

            # Scatter the (jittered) data points
            ax.scatter(
                x_plot, y_plot,
                s=MARKER_SIZE,
                alpha=MARKER_ALPHA,
                color=MARKER_COLOR
            )

            # If there are at least 2 points, compute and plot regression
            if len(x_data) >= 2:
                # Fit OLS model
                X = sm.add_constant(x_data)
                model = sm.OLS(y_data, X).fit()
                p_val = model.f_pvalue
                n_val = len(x_data)
                r_val = np.corrcoef(x_data, y_data)[0,1]
                try:
                    bf_val = pg.bayesfactor_pearson(r_val, n_val)
                except:
                    bf_val = np.nan

                # Compute regression predictions and confidence interval
                # We'll use the buffered x-axis limits below for prediction.
                # For now, we determine new_x based on the current axis limits.
                current_xlim = ax.get_xlim()
                new_x = np.linspace(current_xlim[0], current_xlim[1], 100)
                new_X = sm.add_constant(new_x)
                prediction = model.get_prediction(new_X)
                pred_summary = prediction.summary_frame(alpha=0.05)
                predicted_line = pred_summary["mean"]
                ci_lower = pred_summary["mean_ci_lower"]
                ci_upper = pred_summary["mean_ci_upper"]

                # Plot a solid regression line and fill the confidence band
                ax.plot(new_x, predicted_line, color=REG_COLOR, linewidth=REG_LINE_WIDTH, linestyle="-")
                ax.fill_between(new_x, ci_lower, ci_upper, color=REG_COLOR, alpha=REG_FILL_ALPHA)

                stats_text = (
                    f"r={r_val:.3f}\n"
                    f"p={p_val:.1e}\n"
                    f"BF={bf_val:.2f}\n"
                    f"n={n_val}"
                )
                ax.text(
                    0.05, 0.95, stats_text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=AXIS_LABEL_FONT_SIZE - 2,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
                )

            # Axis buffer approach (using jittered data)
            x_min, x_max = x_plot.min(), x_plot.max()
            y_min, y_max = y_plot.min(), y_plot.max()
            x_range = x_max - x_min
            y_range = y_max - y_min

            if math.isclose(x_min, x_max, rel_tol=1e-9):
                x_min -= 1e-6
                x_max += 1e-6
            if math.isclose(y_min, y_max, rel_tol=1e-9):
                y_min -= 1e-6
                y_max += 1e-6

            x_buffer = x_range * AXIS_BUFFER_RATIO
            y_buffer = y_range * AXIS_BUFFER_RATIO

            x_min_buf = round_down(x_min - x_buffer, DECIMALS_FOR_TICKS)
            x_max_buf = round_up(x_max + x_buffer, DECIMALS_FOR_TICKS)
            y_min_buf = round_down(y_min - y_buffer, DECIMALS_FOR_TICKS)
            y_max_buf = round_up(y_max + y_buffer, DECIMALS_FOR_TICKS)

            ax.set_xlim(x_min_buf, x_max_buf)
            ax.set_ylim(y_min_buf, y_max_buf)

            # Build ticks (exactly 5)
            xticks = np.linspace(x_min_buf, x_max_buf, N_MAJOR_TICKS)
            yticks = np.linspace(y_min_buf, y_max_buf, N_MAJOR_TICKS)
            xticks = [round(v, DECIMALS_FOR_TICKS) for v in xticks]
            yticks = [round(v, DECIMALS_FOR_TICKS) for v in yticks]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.set_xlabel("Rainbow Random $\\psi_{\\theta}$", fontsize=AXIS_LABEL_FONT_SIZE)
            ax.set_ylabel(f"List Sort Form {form_num} $\\psi_{{\\theta}}$", fontsize=AXIS_LABEL_FONT_SIZE)
            ax.grid(False)
            _style_spines_and_ticks(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "Fig10.png"
    pdf_path = output_dir / "Fig10.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved correlation figure to:\n  {png_path}\n  {pdf_path}")

def _style_spines_and_ticks(ax):
    """Aesthetic styling: black spines, outward ticks, etc."""
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(SPINE_LINEWIDTH)
        ax.spines[spine].set_color("black")
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        colors="black",
        labelsize=TICK_LABEL_FONT_SIZE
    )

##############################################################################
# main()
##############################################################################
def main():
    warnings.filterwarnings("ignore")

    data_dir = project_root / "data"
    structures_dir = data_dir / "structures"
    manager = GameDataManager.load_from_pickle(structures_dir / "merged_structure.pkl")
    print(f"Loaded merged structure with {len(manager.players)} participants.")

    build_modes = ["D2_3COLOR"]  # or more if you want
    figs_dir_corr = project_root / "manuscript-1-figures" / "Fig10"

    correlate_bm_vs_lswm_first_bm(manager, build_modes, figs_dir_corr)
    print("\nDone!")

if __name__ == "__main__":
    main()
