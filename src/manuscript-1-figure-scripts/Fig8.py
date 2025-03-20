#!/usr/bin/env python3

import sys
from pathlib import Path
import warnings
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# NEW: For ICC and potential BF
import pingouin as pg

# -----------------------------------------------------------------------------
# Add project root to path
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# If you have a data structure manager
from src.data_structures.game_data import GameDataManager, BarnyardBlastTrial, NetherKnightTrial, DoorDecipherTrial, ParrotSanctuaryTrial

##############################################################################
# PLOTTING PARAMETERS (matching sample script aesthetics)
##############################################################################
FIG_SIZE = 5.0
SUPER_TITLE_FONT_SIZE = 22
SUPER_TITLE = "Test-Retest Reliability of PixelDOPA Game Response Times"
SUPTITLE_Y = .95

MARKER_SIZE = 150.0
MARKER_ALPHA = 0.5
MARKER_COLOR = "steelblue"

REG_LINE_WIDTH = 3.5
REG_COLOR = "darkorange"

AXIS_LABEL_FONT_SIZE = 18
TICK_LABEL_FONT_SIZE = 16
TICK_LENGTH = 6
TICK_WIDTH = 1
BOTTOM_XLABEL_Y = 0.025
LEFT_YLABEL_X   = 0.045

N_MAJOR_TICKS = 5

SPINE_LINEWIDTH = 1

USE_RESPONSE_TIMES = True

##############################################################################
# HELPER: Convert underscores to Camel Case
##############################################################################
def convert_to_camel_case(game_name: str) -> str:
    if not game_name:
        return game_name
    parts = game_name.split("_")
    return " ".join(word.capitalize() for word in parts)

##############################################################################
# EXISTING HELPER FUNCTIONS
##############################################################################
def _extract_trial_rt(trial, use_reaction_time: bool = USE_RESPONSE_TIMES) -> float:
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

def load_config(config_path: Path) -> Dict[str, float]:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def compute_log_mad_threshold(rts: np.ndarray, multiplier: float = 3.0) -> float:
    rts = rts[np.isfinite(rts) & (rts > 0)]
    if len(rts) == 0:
        return None
    log_rts = np.log(rts)
    median_log = np.median(log_rts)
    mad_log = np.median(np.abs(log_rts - median_log))
    threshold_log = median_log + multiplier * mad_log
    threshold = np.exp(threshold_log)
    return threshold

def compute_thresholds_for_games(manager: GameDataManager, games: List[str], max_excessive_rt: float, mad_multiplier: float) -> Dict[str, float]:
    game_rt_data = {g: [] for g in games}
    for player in manager.players.values():
        for game_name, game in player.games.items():
            if game_name in game_rt_data:
                for session in game.sessions.values():
                    for trial in session.trials:
                        rt_val = _extract_trial_rt(trial)
                        if (np.isfinite(rt_val) and rt_val > 0 and rt_val <= max_excessive_rt):
                            game_rt_data[game_name].append(rt_val)
    game_thresholds = {}
    for game_name, rts in game_rt_data.items():
        rts_array = np.array(rts, dtype=float)
        thresh = compute_log_mad_threshold(rts_array, multiplier=mad_multiplier)
        game_thresholds[game_name] = thresh
    return game_thresholds

##############################################################################
# NEW HELPER: Count valid trials in a session based on the raw criteria.
##############################################################################
def count_valid_trials(session, config):
    count = 0
    for t in session.trials:
        rt_val = _extract_trial_rt(t)
        if np.isfinite(rt_val) and rt_val > 0 and rt_val <= config['max_excessive_rt']:
            count += 1
    return count

##############################################################################
# UPDATED FUNCTION: filter_player_game_sessions
#   Now excludes sessions that have fewer than min_samples valid trials,
#   not just after outlier filtering.
##############################################################################
def filter_player_game_sessions(
    manager: GameDataManager,
    pid: str,
    game_name: str,
    threshold: float,
    max_outlier_pct: float,
    max_excessive_rt: float,
    min_samples: int
) -> List[float]:
    player = manager.get_player(pid)
    if not player:
        return None
    game = player.get_game(game_name)
    if not game or not game.sessions:
        return None

    all_filtered_trials = []
    session_exclusions = []

    for session_id, session in game.sessions.items():
        session_rts_raw = []
        for t in session.trials:
            rt_val = _extract_trial_rt(t)
            if np.isfinite(rt_val) and rt_val > 0:
                session_rts_raw.append(rt_val)

        # Drop session if raw valid trials are fewer than min_samples
        if len(session_rts_raw) < min_samples:
            session_exclusions.append((session_id, f"Fewer than min_samples ({min_samples}) in raw session"))
            continue

        excessive_outliers = [rt for rt in session_rts_raw if rt > max_excessive_rt]
        excessive_outlier_pct = (len(excessive_outliers) / len(session_rts_raw)) * 100

        if excessive_outlier_pct > max_outlier_pct:
            session_exclusions.append((session_id, f"Excluded: excessive outlier pct = {excessive_outlier_pct:.2f}% (> {max_outlier_pct}%)"))
            continue
        else:
            session_rts_after_excessive = [rt for rt in session_rts_raw if rt <= max_excessive_rt]

        if len(session_rts_after_excessive) == 0:
            session_exclusions.append((session_id, "All trials excessive after Step 1"))
            continue

        if threshold is not None and threshold > 0:
            threshold_outliers = [rt for rt in session_rts_after_excessive if rt > threshold]
            threshold_outlier_pct = (len(threshold_outliers) / len(session_rts_after_excessive)) * 100
        else:
            threshold_outlier_pct = 0

        if threshold_outlier_pct > max_outlier_pct:
            session_exclusions.append((session_id, f"Excluded: threshold outlier pct = {threshold_outlier_pct:.2f}% (> {max_outlier_pct}%)"))
            continue
        else:
            session_rts_final = [rt for rt in session_rts_after_excessive if rt <= threshold]
            if len(session_rts_final) == 0:
                session_exclusions.append((session_id, "All trials exceeded threshold in Step 2"))
                continue

            if len(session_rts_final) < min_samples:
                session_exclusions.append((session_id, f"Fewer than min_samples ({min_samples}) remain after filtering"))
                continue

            all_filtered_trials.extend(session_rts_final)

    if len(session_exclusions) > 0:
        print(f"DEBUG: For PID {pid}, Game {game_name}, the following sessions were excluded or dropped:")
        for sid, reason in session_exclusions:
            print(f"   Session ID {sid} excluded: {reason}")

    if len(all_filtered_trials) == 0:
        return None
    return all_filtered_trials

##############################################################################
# UPDATED FUNCTION: create_test_retest_figure_with_outlier_exclusion
#   Now only selects sessions that meet the min_samples requirement.
##############################################################################
def create_test_retest_figure_with_outlier_exclusion(manager: GameDataManager, output_dir: Path, config: Dict[str, Any]) -> None:
    games_of_interest = ['NETHER_KNIGHT', 'DOOR_DECIPHER', 'BARNYARD_BLAST']
    use_incomplete_sessions = True

    # Compute thresholds
    game_thresholds = compute_thresholds_for_games(
        manager,
        games_of_interest,
        max_excessive_rt=config['max_excessive_rt'],
        mad_multiplier=config['trial_mad_multiplier']
    )
    
    def filter_session_trials(session, threshold, pid, game, manager, config):
        session_rts = []
        for t in session.trials:
            rt_val = _extract_trial_rt(t)
            if np.isfinite(rt_val) and rt_val > 0 and rt_val <= config['max_excessive_rt']:
                session_rts.append(rt_val)
        if len(session_rts) < config['min_samples']:
            return None
        outliers = [rt for rt in session_rts if rt > threshold]
        outlier_pct = (len(outliers) / len(session_rts)) * 100 if session_rts else 0
        if outlier_pct > config['max_outlier_pct']:
            return None
        session_rts_final = [rt for rt in session_rts if rt <= threshold]
        if len(session_rts_final) < config['min_samples']:
            return None
        return session_rts_final

    # Only select sessions that meet min_samples.
    # We use our new count_valid_trials() helper to filter sessions early.
    test_retest_data = {game: [] for game in games_of_interest}
    
    for pid, player in manager.players.items():
        for game in games_of_interest:
            game_data = player.get_game(game)
            if game_data is None:
                continue

            if not use_incomplete_sessions:
                possible_sessions = [
                    s for s in game_data.sessions.values() 
                    if s.metadata.get('finished', False) and count_valid_trials(s, config) >= config['min_samples']
                ]
                possible_sessions.sort(key=lambda s: s.timestamp)
                if len(possible_sessions) < 2:
                    continue
                session_1, session_2 = possible_sessions[0], possible_sessions[1]
            else:
                completed_only = [
                    s for s in game_data.sessions.values() 
                    if s.metadata.get('finished', False) and count_valid_trials(s, config) >= config['min_samples']
                ]
                completed_only.sort(key=lambda s: s.timestamp)
                
                if len(completed_only) >= 2:
                    session_1, session_2 = completed_only[0], completed_only[1]
                else:
                    all_sess = [s for s in game_data.sessions.values() if count_valid_trials(s, config) >= config['min_samples']]
                    all_sess.sort(key=lambda s: s.timestamp)
                    if len(all_sess) < 2:
                        continue
                    session_1, session_2 = all_sess[0], all_sess[1]
                    
            threshold = game_thresholds.get(game, None)
            if threshold is None:
                continue
            
            session_1_filtered = filter_session_trials(session_1, threshold, pid, game, manager, config)
            session_2_filtered = filter_session_trials(session_2, threshold, pid, game, manager, config)
            
            if session_1_filtered is None or session_2_filtered is None:
                continue
            
            rt1 = np.mean(session_1_filtered) if session_1_filtered else np.nan
            rt2 = np.mean(session_2_filtered) if session_2_filtered else np.nan
            
            if not np.isnan(rt1) and not np.isnan(rt2):
                test_retest_data[game].append((pid, rt1, rt2))
    
    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(FIG_SIZE * 3, FIG_SIZE))
    fig.suptitle(SUPER_TITLE, fontsize=SUPER_TITLE_FONT_SIZE, y=SUPTITLE_Y)

    for ax, game in zip(axes, games_of_interest):
        data = test_retest_data[game]
        if len(data) == 0:
            ax.text(
                0.5, 0.5,
                f"No test-retest data\nfor {convert_to_camel_case(game)}",
                ha='center', va='center', fontsize=12
            )
            ax.set_title(convert_to_camel_case(game), fontsize=AXIS_LABEL_FONT_SIZE)
            ax.grid(False)
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
            ax.set_aspect('equal', adjustable='box')
            continue
        
        df = pd.DataFrame(data, columns=['PID', 'Session1_RT', 'Session2_RT'])
        ax.scatter(df['Session1_RT'], df['Session2_RT'], s=MARKER_SIZE, alpha=MARKER_ALPHA, color=MARKER_COLOR)
        combined_vals = np.concatenate([df["Session1_RT"].values, df["Session2_RT"].values])
        val_min = np.min(combined_vals)
        val_max = np.max(combined_vals)
        if np.isclose(val_min, val_max):
            val_min -= 1e-6
            val_max += 1e-6
        # Compute a 5% buffer on the data range
        range_val = val_max - val_min
        buffer = 0.05 * range_val
        axis_min = round(val_min - buffer, 1)
        axis_max = round(val_max + buffer, 1)

        # Plot the dotted line of equality using the buffered limits
        ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle=':', color='grey')

        sns.regplot(
            x="Session1_RT",
            y="Session2_RT",
            data=df,
            scatter=False,
            color=REG_COLOR,
            line_kws={'linewidth': REG_LINE_WIDTH},
            ci=None,
            ax=ax,
            truncate=False
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
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
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)

        # Generate ticks and round them to one decimal place
        xticks = np.round(np.linspace(axis_min, axis_max, N_MAJOR_TICKS), 1)
        yticks = np.round(np.linspace(axis_min, axis_max, N_MAJOR_TICKS), 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_title(convert_to_camel_case(game), fontsize=AXIS_LABEL_FONT_SIZE)


        df_long = []
        for _, row in df.iterrows():
            df_long.append({'PID': row['PID'], 'Session': 1, 'RT': row['Session1_RT']})
            df_long.append({'PID': row['PID'], 'Session': 2, 'RT': row['Session2_RT']})
        df_long = pd.DataFrame(df_long)
        icc_result = pg.intraclass_corr(
            data=df_long,
            targets='PID',
            raters='Session',
            ratings='RT'
        )
        icc_row = icc_result.query("Type == 'ICC2'")
        if len(icc_row) == 0:
            icc_val = np.nan
            p_val = np.nan
        else:
            icc_val = icc_row['ICC'].values[0]
            p_val = icc_row['pval'].values[0]
        n = len(df)
        ax.text(
            0.03, 0.95,
            f"ICC = {icc_val:.3f}\np = {p_val:.3e}\nn = {n}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=AXIS_LABEL_FONT_SIZE - 2,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )
    
    fig.subplots_adjust(left=0.08, bottom=0.130, right=0.97, top=0.82, wspace=0.20)
    fig.text(0.5, BOTTOM_XLABEL_Y, "Session 1 Mean RT (s)", ha='center', va='center', fontsize=AXIS_LABEL_FONT_SIZE)
    fig.text(LEFT_YLABEL_X, 0.5, "Session 2 Mean RT (s)", ha='center', va='center', rotation='vertical', fontsize=AXIS_LABEL_FONT_SIZE)

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / 'Fig8.png'
    plt.savefig(png_path, dpi=400, bbox_inches='tight')
    pdf_path = output_dir / 'Fig8.pdf'
    plt.savefig(pdf_path, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"Test-retest figure saved to {png_path}")
    print(f"Test-retest figure also saved as PDF to {pdf_path}")

##############################################################################
# main()
##############################################################################
def main():
    warnings.filterwarnings('ignore')
    data_dir = project_root / 'data'
    structures_dir = data_dir / 'structures'
    config_path = project_root / 'cleaning_config.json'
    config = load_config(config_path)
    manager = GameDataManager.load_from_pickle(structures_dir / 'merged_structure.pkl')
    test_retest_dir = project_root / 'manuscript-1-figures' / 'Fig8'
    create_test_retest_figure_with_outlier_exclusion(manager, test_retest_dir, config)
    print("Done creating test-retest figure with outlier exclusion.")

if __name__ == '__main__':
    sys.exit(main())
