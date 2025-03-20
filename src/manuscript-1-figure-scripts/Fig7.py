#!/usr/bin/env python3
"""
plot_RT_correlations.py

Creates scatterplots comparing reaction times (RT) for paired NIH vs. MC tasks.
Also archives imported data and filtered data with a timestamp, and saves a pickle
of the raw imported data (trial level).
"""

import sys
from pathlib import Path
import warnings
import math
import json
import pickle  # For saving the raw imported data as a pickle file
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pingouin as pg
import statsmodels.api as sm  # for p-values (F-tests)
from datetime import datetime  # For timestamping archived files

# -----------------------------------------------------------------------------
# Add project root to path
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import data structure manager
from src.data_structures.game_data import (
    GameDataManager,
    BarnyardBlastTrial,
    NetherKnightTrial,
    DoorDecipherTrial,
    ParrotSanctuaryTrial,
)

##############################################################################
# GLOBAL PLOTTING PARAMETERS (change these to affect all plots)
##############################################################################
FIG_SIZE = 5.0             # Each subplot will be FIG_SIZE x FIG_SIZE inches
MARKER_SIZE = 150.0        # Marker (dot) size
MARKER_ALPHA = 0.5         # Transparency for scatter dots
MARKER_COLOR = "steelblue" # Color for scatter dots

REG_LINE_WIDTH = 3.5       # Line width for regression line
REG_COLOR = "darkorange"   # Color for regression line
REG_FILL_ALPHA = 0.2       # Transparency for regression confidence interval fill

INFO_TEXT_FONT_SIZE = 12   # Font size for annotation text (r, p, BF, etc.)
AXIS_LABEL_FONT_SIZE = 14  # Font size for axis labels
TICK_LABEL_FONT_SIZE = 16  # Font size for tick labels
SUBPLOT_TITLE_FONT_SIZE = 0  # Font size for each subplot title (unused)
SUPER_TITLE_FONT_SIZE = 22    # Font size for overall figure title

N_MAJOR_TICKS = 5          # Number of major ticks for each axis
TICK_LENGTH = 6            # Tick mark length
TICK_WIDTH = 1             # Tick mark width

DECIMALS_FOR_TICKS = 1     # Number of decimal places for tick labels
AXIS_BUFFER_RATIO = 0.05   # Buffer ratio (5%) applied to each axis's range
SUPER_TITLE = "Paired Gaze Response Times for NIH Toolbox Tasks vs PixelDOPA Games"  # Overall figure title
SUPTITLE_Y = 1.02          # Vertical position for the super title
INFO_TEXT_Y = .95

USE_RESPONSE_TIMES = False

ARCHIVE_DATA = False
##############################################################################
# HELPER FUNCTIONS
##############################################################################
def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def _extract_trial_rt(trial, use_reaction_time: bool = USE_RESPONSE_TIMES) -> float:
    """
    Extract the reaction time (RT) from a trial.
    
    If use_reaction_time is False, attempt to use an alternate time metric.
    """
    if not use_reaction_time:
        if isinstance(trial, BarnyardBlastTrial):
            tval = trial.additional_info.get("time_to_target_chosen")
            return tval if tval is not None else float("nan")
        elif isinstance(trial, NetherKnightTrial):
            sval = trial.additional_info.get("time_to_skeleton_chosen")
            return sval if sval is not None else float("nan")
        elif isinstance(trial, DoorDecipherTrial):
            dval = trial.additional_info.get("time_to_door_fixation")
            return dval if dval is not None else float("nan")
        elif isinstance(trial, ParrotSanctuaryTrial):
            dval = trial.additional_info.get("time_to_target")
            return dval if dval is not None else float("nan")
        else:
            return trial.reaction_time
    else:
        return trial.reaction_time


def build_combined_df_from_manager(
    manager: GameDataManager, filter_game_names: List[str] = None
) -> pd.DataFrame:
    """
    Build a combined DataFrame containing aggregated game statistics for each player.
    """
    rows = []
    for pid, player in manager.players.items():
        for game_name, game_data in player.games.items():
            if filter_game_names and game_name not in filter_game_names:
                continue
            stats = game_data.get_aggregate_stats()
            row = {
                "PID": pid,
                "Game": game_name,
                "mean_rt": stats["mean_rt"],
                "mean_accuracy": stats["mean_accuracy"],
                "std_rt": stats["std_rt"],
                "total_trials": stats["total_trials"],
                "total_sessions": stats["total_sessions"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def compute_log_mad_threshold(rts: np.ndarray, multiplier: float = 4.0) -> float:
    """
    Compute the log-based MAD threshold for outlier rejection.
    
    Returns the exponentiated threshold.
    """
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) == 0:
        return None
    log_rts = np.log(rts)
    median_log = np.median(log_rts)
    mad_log = np.median(np.abs(log_rts - median_log))
    threshold_log = median_log + multiplier * mad_log
    threshold = np.exp(threshold_log)
    return threshold


def compute_thresholds_for_games(
    manager: GameDataManager, games: List[str], max_excessive_rt: float, mad_multiplier: float
) -> Dict[str, float]:
    """
    Compute per-game RT thresholds using a log-MAD method.
    """
    game_rt_data = {g: [] for g in games}
    for player in manager.players.values():
        for game_name, game in player.games.items():
            if game_name in game_rt_data:
                for session in game.sessions.values():
                    for trial in session.trials:
                        rt_val = _extract_trial_rt(trial)
                        if np.isfinite(rt_val) and rt_val > 0 and rt_val <= max_excessive_rt:
                            game_rt_data[game_name].append(rt_val)

    game_thresholds = {}
    for game_name, rts in game_rt_data.items():
        rts_array = np.array(rts, dtype=float)
        thresh = compute_log_mad_threshold(rts_array, multiplier=mad_multiplier)
        game_thresholds[game_name] = thresh
    return game_thresholds


def filter_player_game_sessions(
    manager: GameDataManager,
    pid: str,
    game_name: str,
    threshold: float,
    max_outlier_pct: float,
    max_excessive_rt: float,
) -> Tuple[List[float], int, int, int, int]:
    """
    Filter trials for a given participant and game.
    
    Returns:
        final_rts: Filtered RT values.
        total_raw_count: Total valid trials before filtering.
        kept_count: Number of trials remaining after filtering.
        total_sessions_count: Total number of sessions.
        lost_sessions_count: Number of sessions excluded.
    """
    player = manager.get_player(pid)
    if not player:
        return None, 0, 0, 0, 0
    game = player.get_game(game_name)
    if not game or not game.sessions:
        return None, 0, 0, 0, 0

    total_sessions_count = len(game.sessions)
    lost_sessions_count = 0
    all_filtered_trials = []
    session_exclusions = []
    total_raw_count = 0

    for session_id, session in game.sessions.items():
        session_rts_raw = []
        for t in session.trials:
            rt_val = _extract_trial_rt(t)
            if np.isfinite(rt_val) and rt_val > 0:
                session_rts_raw.append(rt_val)
        total_raw_count += len(session_rts_raw)

        if len(session_rts_raw) == 0:
            session_exclusions.append((session_id, "No valid trials (raw)"))
            continue

        # Step 1: Exclude excessive RT outliers
        excessive_outliers = [rt for rt in session_rts_raw if rt > max_excessive_rt]
        excessive_outlier_pct = (len(excessive_outliers) / len(session_rts_raw) * 100) if session_rts_raw else 0
        if excessive_outlier_pct > max_outlier_pct:
            session_exclusions.append(
                (session_id, f"Excluded: excessive outlier pct = {excessive_outlier_pct:.2f}%")
            )
            continue

        session_rts_after_excessive = [rt for rt in session_rts_raw if rt <= max_excessive_rt]
        if len(session_rts_after_excessive) == 0:
            session_exclusions.append((session_id, "All trials excessive after Step 1"))
            continue

        # Step 2: Threshold-based outlier removal (MAD threshold)
        if threshold is not None and threshold > 0:
            threshold_outliers = [rt for rt in session_rts_after_excessive if rt > threshold]
            threshold_outlier_pct = (len(threshold_outliers) / len(session_rts_after_excessive) * 100)
        else:
            threshold_outlier_pct = 0

        if threshold_outlier_pct > max_outlier_pct:
            session_exclusions.append(
                (session_id, f"Excluded: threshold outlier pct = {threshold_outlier_pct:.2f}%")
            )
            continue
        else:
            session_rts_final = [rt for rt in session_rts_after_excessive if rt <= threshold]
            if len(session_rts_final) == 0:
                session_exclusions.append((session_id, "All trials exceeded threshold in Step 2"))
                continue
            all_filtered_trials.extend(session_rts_final)

    lost_sessions_count = len(session_exclusions)

    if session_exclusions:
        print(f"DEBUG: For PID {pid}, Game {game_name}, the following sessions were excluded/dropped:")
        for sid, reason in session_exclusions:
            print(f"   Session ID {sid} excluded: {reason}")

    if len(all_filtered_trials) == 0:
        return None, total_raw_count, 0, total_sessions_count, lost_sessions_count

    return all_filtered_trials, total_raw_count, len(all_filtered_trials), total_sessions_count, lost_sessions_count


def filter_rt_all_games_once(
    manager: GameDataManager,
    combined_df: pd.DataFrame,
    games: List[str],
    mad_multiplier: float,
    max_excessive_rt: float,
    max_outlier_pct: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter RTs for all games once, and summarize the number of lost trials/sessions.
    
    Returns:
        df_allgames_rt: Wide table of mean RT by participant × game.
        summary_df: Summary DataFrame for lost trials/sessions per game.
    """
    print("\n[Clean-Once] Computing per-game thresholds (MAD-based) for RT ...")
    game_thresholds = compute_thresholds_for_games(
        manager=manager,
        games=games,
        max_excessive_rt=max_excessive_rt,
        mad_multiplier=mad_multiplier,
    )
    print(game_thresholds)

    # Track exclusion stats per game
    excluded_stats = {
        g: {"trials_total": 0, "trials_kept": 0, "sessions_total": 0, "sessions_lost": 0}
        for g in games
    }

    df_dict = {}
    all_pids = combined_df["PID"].unique()
    for pid in all_pids:
        df_dict[pid] = {}

    for g in games:
        threshold_g = game_thresholds.get(g, None)
        for pid in all_pids:
            final_rts, total_raw_count, kept_count, total_sess_count, lost_sess_count = filter_player_game_sessions(
                manager, pid, g, threshold_g, max_outlier_pct, max_excessive_rt
            )

            if total_raw_count > 0:
                excluded_stats[g]["trials_total"] += total_raw_count
                excluded_stats[g]["trials_kept"] += kept_count
                excluded_stats[g]["sessions_total"] += total_sess_count
                excluded_stats[g]["sessions_lost"] += lost_sess_count

            df_dict[pid][f"{g}_RT"] = np.nanmean(final_rts) if final_rts and len(final_rts) > 0 else np.nan

    columns = ["PID"] + [f"{g}_RT" for g in games]
    rows = []
    for pid, val_map in df_dict.items():
        row = {"PID": pid}
        for c in columns[1:]:
            row[c] = val_map.get(c, np.nan)
        rows.append(row)

    df_allgames_rt = pd.DataFrame(rows, columns=columns)

    # Build a summary DataFrame from excluded stats
    summary_rows = []
    for game_name, counts in excluded_stats.items():
        trials_total = counts["trials_total"]
        trials_kept = counts["trials_kept"]
        sessions_total = counts["sessions_total"]
        sessions_lost = counts["sessions_lost"]

        trials_lost = trials_total - trials_kept
        trial_lost_pct = (trials_lost / trials_total * 100.0) if trials_total > 0 else 0
        session_lost_pct = (sessions_lost / sessions_total * 100.0) if sessions_total > 0 else 0

        summary_rows.append({
            "Game": game_name,
            "TrialsTotal": trials_total,
            "TrialsLost": trials_lost,
            "TrialsLostPct": f"{trial_lost_pct:.2f}",
            "SessionsTotal": sessions_total,
            "SessionsLost": sessions_lost,
            "SessionsLostPct": f"{session_lost_pct:.2f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    return df_allgames_rt, summary_df


def build_accuracy_wide_table(
    combined_df: pd.DataFrame, games: List[str]
) -> pd.DataFrame:
    """
    Build a wide table for accuracy, with one column per game.
    """
    df_sub = combined_df[combined_df["Game"].isin(games)].copy()
    pivot_data = {}
    for pid in df_sub["PID"].unique():
        pivot_data[pid] = {}

    for _, row in df_sub.iterrows():
        pid = row["PID"]
        g = row["Game"]
        acc = row["mean_accuracy"]
        pivot_data[pid][f"{g}_ACC"] = acc

    columns = ["PID"] + [f"{g}_ACC" for g in games]
    rows = []
    for pid, val_map in pivot_data.items():
        row = {"PID": pid}
        for c in columns[1:]:
            row[c] = val_map.get(c, np.nan)
        rows.append(row)

    df_acc = pd.DataFrame(rows, columns=columns)
    return df_acc


def do_final_value_exclusion(df: pd.DataFrame, mad_multiplier: float = 3.0) -> pd.DataFrame:
    """
    For each numeric column in df:
      1) Compute the median and MAD.
      2) Identify values where |x - median|/MAD > mad_multiplier.
      3) Set those cells to NaN.
    """
    df_out = df.copy()
    if "PID" in df_out.columns:
        numeric_cols = df_out.drop(columns=["PID"]).select_dtypes(include=[np.number]).columns
    else:
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        col_data = df_out[col].values
        finite_mask = np.isfinite(col_data)
        if not np.any(finite_mask):
            continue

        median_val = np.median(col_data[finite_mask])
        abs_dev = np.abs(col_data - median_val)
        mad_val = np.median(abs_dev[finite_mask])

        if mad_val == 0 or np.isnan(mad_val):
            continue

        ratio = abs_dev / mad_val
        outlier_mask = ratio > mad_multiplier
        df_out.loc[outlier_mask, col] = np.nan

    return df_out

##############################################################################
# PLOTTING FUNCTION: 1x3 Scatterplot Grid for RT Pairs (NIH vs. MC)
##############################################################################
def create_1x3_scatterplot_rt_pairs(
    df_wide_all: pd.DataFrame,
    mc_tasks: List[str],
    nih_tasks: List[str],
    min_samples: int,
    output_dir: Path,
    # Plotting parameters with default values
    fig_size: float = FIG_SIZE,
    marker_size: float = MARKER_SIZE,
    marker_alpha: float = MARKER_ALPHA,
    marker_color: str = MARKER_COLOR,
    reg_line_width: float = REG_LINE_WIDTH,
    reg_color: str = REG_COLOR,
    reg_fill_alpha: float = REG_FILL_ALPHA,
    info_text_fontsize: int = INFO_TEXT_FONT_SIZE,
    axis_label_fontsize: int = AXIS_LABEL_FONT_SIZE,
    tick_label_fontsize: int = TICK_LABEL_FONT_SIZE,
    subplot_title_fontsize: int = SUBPLOT_TITLE_FONT_SIZE,
    super_title_fontsize: int = SUPER_TITLE_FONT_SIZE,
    decimals_for_ticks: int = DECIMALS_FOR_TICKS,
    axis_buffer_ratio: float = AXIS_BUFFER_RATIO,
    super_title: str = SUPER_TITLE,
    suptitle_y: float = SUPTITLE_Y,
    n_major_ticks: int = N_MAJOR_TICKS,
    tick_length: int = TICK_LENGTH,
    tick_width: int = TICK_WIDTH,
    info_text_y: float = INFO_TEXT_Y,  # New parameter for vertical position of annotation
) -> None:
    """
    Create a 1×3 grid of scatterplots comparing RT for paired NIH and MC tasks.
    
    Parameters:
      ...
      info_text_y : float
          Vertical position (in axes coordinates) for the annotation text. 
          Lower values move the block further down. Default is 0.85.
    """
    if len(nih_tasks) != len(mc_tasks):
        print("Error: The number of NIH tasks must match the number of MC tasks.")
        return

    # Mapping of raw task names to display labels
    task_label_map = {
        "Pattern Comparison Processing Speed": "Pattern Comparison Processing Speed",
        "Dimensional Change Card Sort": "Dimensional Change Card Sort",
        "Flanker Inhibitory Control and Attention": "Flanker Inhibitory Control and Attention",
        "NETHER_KNIGHT": "Nether Knight",
        "DOOR_DECIPHER": "Door Decipher",
        "BARNYARD_BLAST": "Barnyard Blast",
    }

    def round_down(value, decimals=1):
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def round_up(value, decimals=1):
        factor = 10 ** decimals
        return math.ceil(value * factor) / factor

    n_pairs = len(nih_tasks)

    # Create the figure; each subplot is square
    fig, axes = plt.subplots(
        1, n_pairs,
        figsize=(fig_size * n_pairs, fig_size),
        constrained_layout=True,
    )
    if n_pairs == 1:
        axes = [axes]

    for idx, (nih_task, mc_task) in enumerate(zip(nih_tasks, mc_tasks)):
        ax = axes[idx]
        ax.set_box_aspect(1)  # Ensure square axes

        x_col = f"{nih_task}_RT"
        y_col = f"{mc_task}_RT"
        x_label = f"{task_label_map.get(nih_task, nih_task)} RT"
        y_label = f"{task_label_map.get(mc_task, mc_task)} gRT"

        # Subset data for the task pair
        sub_df = df_wide_all.dropna(subset=[x_col, y_col]).copy()
        n_val = len(sub_df)
        if n_val < min_samples:
            ax.text(
                0.5, 0.5,
                f"n={n_val}\nNot enough data",
                ha="center", va="center",
                fontsize=axis_label_fontsize,
            )
            ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
            continue

        # Extract data arrays
        x_arr = sub_df[x_col].values
        y_arr = sub_df[y_col].values

        # Plot scatter points
        ax.scatter(
            x_arr,
            y_arr,
            s=marker_size,
            alpha=marker_alpha,
            color=marker_color,
        )

        # Fit linear regression
        X = sm.add_constant(x_arr)
        model = sm.OLS(y_arr, X).fit()

        # Compute buffered axis limits
        x_min_data, x_max_data = x_arr.min(), x_arr.max()
        y_min_data, y_max_data = y_arr.min(), y_arr.max()
        x_range = x_max_data - x_min_data
        y_range = y_max_data - y_min_data
        x_buffer = x_range * axis_buffer_ratio
        y_buffer = y_range * axis_buffer_ratio

        x_min_buffered = round_down(x_min_data - x_buffer, decimals_for_ticks)
        x_max_buffered = round_up(x_max_data + x_buffer, decimals_for_ticks)
        y_min_buffered = round_down(y_min_data - y_buffer, decimals_for_ticks)
        y_max_buffered = round_up(y_max_data + y_buffer, decimals_for_ticks)

        # Set axis limits and compute evenly spaced ticks
        ax.set_xlim(x_min_buffered, x_max_buffered)
        ax.set_ylim(y_min_buffered, y_max_buffered)
        x_ticks = np.linspace(x_min_buffered, x_max_buffered, n_major_ticks)
        y_ticks = np.linspace(y_min_buffered, y_max_buffered, n_major_ticks)
        x_ticks = [round(t, decimals_for_ticks) for t in x_ticks]
        y_ticks = [round(t, decimals_for_ticks) for t in y_ticks]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Set tick parameters (label size and tick mark properties)
        ax.tick_params(
            labelsize=tick_label_fontsize,
            length=tick_length,
            width=tick_width,
        )

        # Regression line and confidence interval
        new_x = np.linspace(x_min_buffered, x_max_buffered, 100)
        new_X = sm.add_constant(new_x)
        prediction = model.get_prediction(new_X)
        pred_summary = prediction.summary_frame(alpha=0.05)
        predicted_line = pred_summary["mean"]
        ci_lower = pred_summary["mean_ci_lower"]
        ci_upper = pred_summary["mean_ci_upper"]
        ax.plot(new_x, predicted_line, color=reg_color, linewidth=reg_line_width)
        ax.fill_between(new_x, ci_lower, ci_upper, color=reg_color, alpha=reg_fill_alpha)

        # Annotation with statistics
        r_val, p_val = stats.pearsonr(x_arr, y_arr)
        try:
            bf_val = pg.bayesfactor_pearson(r_val, n_val)
        except Exception as e:
            print(f"[DEBUG] BF exception: {repr(e)}")
            bf_val = np.nan
        rmse = np.sqrt(np.mean((y_arr - model.fittedvalues) ** 2)) / n_val  # NRMSE adjusted for n
        annotation = (
            f"r = {r_val:.3f}\n"
            f"p = {p_val:.1e}\n"
            f"BF = {bf_val:.2f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"n = {n_val}"
        )
        ax.text(
            0.025, info_text_y,  # Use the new parameter here instead of the hard-coded 0.95
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=info_text_fontsize,
            bbox=dict(facecolor="none", alpha=0.7, edgecolor="none"),
        )

        ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
        ax.set_ylabel(y_label, fontsize=axis_label_fontsize)

    # Overall figure title with adjustable vertical placement
    fig.suptitle(super_title, fontsize=super_title_fontsize, y=suptitle_y)

    # Save PNG
    png_path = output_dir / 'Fig7.png'
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    
    # Save PDF
    pdf_path = output_dir / 'Fig7.pdf'
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    plt.close()


##############################################################################
# MAIN FUNCTION
##############################################################################
def main():
    warnings.filterwarnings("ignore")

    # Create a timestamp for archiving files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_dir = project_root / "data"
    final_plots_dir = project_root / "manuscript-1-figures" / "Fig7"
    final_plots_dir.mkdir(parents=True, exist_ok=True)

    config_path = project_root / "cleaning_config.json"
    config = load_config(config_path)

    trial_mad_multiplier = config.get("trial_mad_multiplier", 4.0)
    mean_mad_multiplier = config.get("mean_mad_multiplier", 3.0)
    max_excessive_rt = config.get("max_excessive_rt", 10.0)
    max_outlier_pct = config.get("max_outlier_pct", 33.0)
    remove_final_outlier_participants = config.get("remove_final_outlier_participants", True)
    min_samples = config.get("min_samples", 5)

    print("Loading merged GameDataManager from pickle...")
    manager = GameDataManager.load_from_pickle(data_dir / "structures" / "merged_structure.pkl")

    if ARCHIVE_DATA:
        # Archive the raw imported data (pickle) with a timestamp.
        raw_pickle_path = final_plots_dir / f"raw_imported_data_{timestamp}.pkl"
        with open(raw_pickle_path, "wb") as f:
            pickle.dump(manager, f)
        print(f"[ARCHIVE] Raw imported data pickle saved to {raw_pickle_path}")

    # 1) Build combined DataFrame from manager
    combined_df = build_combined_df_from_manager(manager, filter_game_names=None)
    print("Combined DataFrame shape:", combined_df.shape)
    print(combined_df.head())

    if ARCHIVE_DATA:
        # Archive the imported data (CSV) with a timestamp.
        imported_data_path = final_plots_dir / f"imported_data_{timestamp}.csv"
        combined_df.to_csv(imported_data_path, index=False)
        print(f"[ARCHIVE] Imported data saved to {imported_data_path}")

    # 2) Define tasks
    nih_tasks = [
        "Pattern Comparison Processing Speed",
        "Dimensional Change Card Sort",
        "Flanker Inhibitory Control and Attention",
    ]
    mc_tasks = [
        "NETHER_KNIGHT",
        "DOOR_DECIPHER",
        "BARNYARD_BLAST",
    ]
    all_games = nih_tasks + mc_tasks

    # 3) Filter RT at the trial level for these tasks
    df_allgames_rt, summary_df = filter_rt_all_games_once(
        manager=manager,
        combined_df=combined_df,
        games=all_games,
        mad_multiplier=trial_mad_multiplier,
        max_excessive_rt=max_excessive_rt,
        max_outlier_pct=max_outlier_pct,
    )

    # Save summary CSV about lost trials/sessions
    summary_csv_path = final_plots_dir / "lost_trials_sessions_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[INFO] Lost Trials/Sessions summary saved to {summary_csv_path}")

    # 4) Build accuracy wide table and merge with RT table
    df_allgames_acc = build_accuracy_wide_table(combined_df=combined_df, games=all_games)
    df_wide_all = pd.merge(df_allgames_rt, df_allgames_acc, on="PID", how="inner")

    # 5) Optionally perform participant-level outlier removal
    if remove_final_outlier_participants:
        df_wide_all = do_final_value_exclusion(df_wide_all, mad_multiplier=mean_mad_multiplier)

    if ARCHIVE_DATA:
        # Archive the filtered data (CSV) with a timestamp.
        filtered_data_path = final_plots_dir / f"filtered_data_{timestamp}.csv"
        df_wide_all.to_csv(filtered_data_path, index=False)
        print(f"[ARCHIVE] Filtered data saved to {filtered_data_path}")

    # 6) Create 1×3 scatterplots for RT comparisons (paired NIH vs MC tasks)
    create_1x3_scatterplot_rt_pairs(
        df_wide_all=df_wide_all,
        mc_tasks=mc_tasks,
        nih_tasks=nih_tasks,
        min_samples=min_samples,
        output_dir=final_plots_dir,
    )

    print("\nDone! Outputs include:")
    print(" - A CSV summary of lost trials/sessions per game")
    print(" - A 1×3 scatterplot comparing RT (NIH vs MC tasks)")
    print(" - Archived imported data and filtered data, and a pickle of the raw imported data.")


if __name__ == "__main__":
    main()
