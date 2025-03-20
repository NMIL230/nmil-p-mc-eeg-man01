# game_data.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import pickle
from collections import defaultdict

@dataclass
class Trial:
    """Represents a single trial in a game session."""
    start_time: float
    end_time: float
    duration: float
    success: bool
    trial_type: str
    difficulty: Optional[int] = None

    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def reaction_time(self) -> float:
        """Calculate reaction time in seconds (example: 1 tick = 1/20 second)."""
        return self.duration / 20  # or however your engine’s “ticks” map to real time

@dataclass
class BarnyardBlastTrial(Trial):
    """
    Specialized Trial for BARNYARD_BLAST that includes a time_to_target metric.
    time_to_target: The difference (in game ticks) from the trial start_time to 
                    the first time the ray_trace_block == 'target'.
    """
    time_to_target: Optional[float] = None

@dataclass
class NetherKnightTrial(Trial):
    """
    Specialized Trial for NETHER_KNIGHT that includes a time_to_skeleton metric.
    time_to_skeleton: The difference (in game ticks) from the trial start_time 
                      to the first time 'entity' == 'skeleton'.
    """
    time_to_skeleton: Optional[float] = None

@dataclass
class DoorDecipherTrial(Trial):
    """
    Specialized Trial for DOOR_DECIPHER that includes a time_to_dummy_box metric.
    time_to_dummy_box: The difference (in game ticks) from the trial start_time
                       to the first time the line from player_pos -> block_hit_pos
                       intersects the final bounding box.
    """
    time_to_dummy_box: Optional[float] = None

@dataclass
class ParrotSanctuaryTrial(Trial):
    """
    Specialized Trial for PARROT_SANCTUARY that tracks time_to_target.
    Just like BarnyardBlastTrial does with time_to_target.
    """
    time_to_target: Optional[float] = None

@dataclass
class BuildLevel:
    """Represents a single BUILD_MASTER level"""
    difficulty: int
    mode: str  # 2D or 3D mode
    attempts: List[Trial] = field(default_factory=list)
    passed: bool = False
    
    @property
    def total_time(self) -> float:
        """Calculate total time spent on level"""
        return sum(t.reaction_time for t in self.attempts)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for level"""
        return sum(1 for t in self.attempts if t.success) / len(self.attempts) if self.attempts else 0
    
    @property
    def avg_attempt_time(self) -> float:
        """Calculate average time per attempt"""
        return self.total_time / len(self.attempts) if self.attempts else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert level data to dictionary"""
        return {
            'difficulty': self.difficulty,
            'mode': self.mode,
            'n_attempts': len(self.attempts),
            'passed': self.passed,
            'total_time': self.total_time,
            'success_rate': self.success_rate * 100,
            'avg_attempt_time': self.avg_attempt_time,
            'first_success_attempt': next((i+1 for i, t in enumerate(self.attempts) if t.success), None)
        }

@dataclass
class BuildMasterData:
    """Container for BUILD_MASTER game data."""
    levels: Dict[Tuple[str, int], BuildLevel] = field(default_factory=dict)
    
    @property
    def highest_completed(self) -> Dict[str, int]:
        completed = {}
        for (mode, diff), level in self.levels.items():
            if level.passed:
                completed[mode] = max(completed.get(mode, 0), diff)
        return completed
    
    def add_attempt(self, trial: Trial) -> None:
        """
        Add an attempt to the appropriate level. 
        We'll also set 'trial.difficulty' so it can be used for session-based grouping.
        """
        mode = trial.additional_info.get("game_mode", "UNKNOWN")
        difficulty = trial.additional_info.get("difficulty", 0)
        trial.difficulty = difficulty  # <-- set the top-level field

        key = (mode, difficulty)
        if key not in self.levels:
            self.levels[key] = BuildLevel(difficulty=difficulty, mode=mode)
        
        self.levels[key].attempts.append(trial)
        if trial.success:
            self.levels[key].passed = True
    
    def get_level_progression(self, mode: str) -> List[Dict]:
        mode_levels = sorted(
            [(diff, level) for (m, diff), level in self.levels.items() if m == mode],
            key=lambda x: x[0]
        )
        return [level.to_dict() for _, level in mode_levels]

@dataclass
class GameSession:
    """Represents a single gaming session"""
    session_id: str
    timestamp: datetime
    trials: List[Trial] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the session"""
        self.trials.append(trial)
    
    def get_accuracy(self) -> float:
        """Calculate session accuracy"""
        if not self.trials:
            return 0.0
        return sum(trial.success for trial in self.trials) / len(self.trials) * 100
    
    def get_mean_rt(self) -> float:
        """Calculate mean reaction time for session"""
        if not self.trials:
            return 0.0
        return np.mean([trial.reaction_time for trial in self.trials])
    
    def get_filtered_trials(self, trial_type: Optional[str] = None, 
                          min_rt: Optional[float] = None, 
                          max_rt: Optional[float] = None) -> List[Trial]:
        """
        Get trials filtered by type and reaction time bounds
        """
        filtered_trials = self.trials
        
        if trial_type is not None:
            filtered_trials = [t for t in filtered_trials if t.trial_type == trial_type]
            
        if min_rt is not None:
            filtered_trials = [t for t in filtered_trials if t.reaction_time >= min_rt]
            
        if max_rt is not None:
            filtered_trials = [t for t in filtered_trials if t.reaction_time <= max_rt]
            
        return filtered_trials

########################################
# LSWM-Level & Container
########################################
@dataclass
class LswmLevel:
    """
    Represents a single difficulty level for LSWM 
    (e.g. sequence length = 2, 3, 4, etc.).
    """
    difficulty: int
    attempts: List["Trial"] = field(default_factory=list)
    passed: bool = False  # You can define "passed" however you like

    @property
    def success_rate(self) -> float:
        """Calculate success rate for this difficulty."""
        if not self.attempts:
            return 0.0
        successes = sum(1 for t in self.attempts if t.success)
        return successes / len(self.attempts)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "total_attempts": self.total_attempts,
            "success_rate": self.success_rate * 100,
            "passed": self.passed
        }

@dataclass
class LswmData:
    """Container for LSWM data, storing attempts by integer difficulty."""
    levels: Dict[int, LswmLevel] = field(default_factory=dict)

    def add_attempt(self, trial: Trial, difficulty: int) -> None:
        """
        Add the given trial to the appropriate LSWM difficulty level.
        We'll also set trial.difficulty so we can do session-based grouping.
        """
        trial.difficulty = difficulty  # set top-level
        if difficulty not in self.levels:
            self.levels[difficulty] = LswmLevel(difficulty=difficulty)

        self.levels[difficulty].attempts.append(trial)
        if trial.success:
            self.levels[difficulty].passed = True

    def get_success_rate_by_difficulty(self) -> Dict[int, float]:
        return {lvl.difficulty: lvl.success_rate for lvl in self.levels.values()}

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for diff, lvl in self.levels.items():
            rows.append(lvl.to_dict())
        return pd.DataFrame(rows)
    
def _lswm_extract_difficulty(trial: "Trial") -> Optional[int]:
    """
    Helper to parse a trial’s item ID (e.g., "LSWM_1List_Live_Item_C")
    and map it to a sequence length (2, 3, 4, etc.).
    """
    item_id = trial.additional_info.get("ItemID", "")
    if not item_id.startswith("LSWM_"):
        return None

    # Example: last char indicates form => length map
    sequence_length_map = {
        'A': 2, 'B': 2, 'M': 2, 'N': 2,
        'C': 3, 'D': 3, 'O': 3, 'P': 3,
        'E': 4, 'F': 4, 'Q': 4, 'R': 4,
        'G': 5, 'H': 5, 'S': 5, 'T': 5,
        'I': 6, 'J': 6, 'U': 6, 'V': 6,
        'K': 7, 'L': 7, 'W': 7, 'X': 7
    }

    last_char = item_id[-1:]  # e.g. "C" in "LSWM_1List_Live_Item_C"
    return sequence_length_map.get(last_char, None)

@dataclass
class GameData:
    """Represents all sessions for a specific game"""
    game_name: str
    sessions: Dict[str, "GameSession"] = field(default_factory=dict)
    build_master_data: Optional["BuildMasterData"] = None

    # NEW: for LSWM
    lswm_data: Optional["LswmData"] = None

    def add_session(self, session: "GameSession") -> None:
        """Add a session to the game."""
        self.sessions[session.session_id] = session

        # Special handling for BUILD_MASTER (unchanged)
        if self.game_name == 'BUILD_MASTER':
            if self.build_master_data is None:
                self.build_master_data = BuildMasterData()
            for trial in session.trials:
                self.build_master_data.add_attempt(trial)

        # NEW: special handling for LSWM
        if self.game_name == "List Sorting Working Memory":
            if self.lswm_data is None:
                self.lswm_data = LswmData()
            for trial in session.trials:
                difficulty = _lswm_extract_difficulty(trial)
                if difficulty is not None:
                    self.lswm_data.add_attempt(trial, difficulty=difficulty)
    
    def get_aggregate_stats(self, min_rt: Optional[float] = None,
                          max_rt: Optional[float] = None) -> Dict[str, float]:
        """Calculate aggregate statistics across all sessions"""
        all_trials = []
        for session in self.sessions.values():
            all_trials.extend(session.get_filtered_trials(min_rt=min_rt, max_rt=max_rt))
        
        if not all_trials:
            return {
                'mean_accuracy': 0.0,
                'mean_rt': 0.0,
                'std_rt': 0.0,
                'total_trials': 0,
                'total_sessions': 0
            }
            
        reaction_times = [trial.reaction_time for trial in all_trials]
        
        stats = {
            'mean_accuracy': np.mean([trial.success for trial in all_trials]) * 100,
            'mean_rt': np.mean(reaction_times),
            'std_rt': np.std(reaction_times) if len(reaction_times) > 1 else 0.0,
            'total_trials': len(all_trials),
            'total_sessions': len(self.sessions)
        }
        
        # Add BUILD_MASTER specific stats if applicable
        if self.game_name == 'BUILD_MASTER' and self.build_master_data:
            highest_completed = self.build_master_data.highest_completed
            for mode, level in highest_completed.items():
                stats[f'highest_completed_{mode}'] = level

        # Also, if you want to store "highest completed difficulty" for LSWM:
        if self.game_name == "List Sorting Working Memory" and self.lswm_data:
            if self.lswm_data.levels:
                max_diff = max(self.lswm_data.levels.keys())
                stats["lswm_highest_difficulty"] = max_diff
        
        return stats
    
    def get_trials_by_type(self) -> Dict[str, List[Trial]]:
        """Group all trials by trial type"""
        trial_types = defaultdict(list)
        for session in self.sessions.values():
            for trial in session.trials:
                trial_types[trial.trial_type].append(trial)
        return dict(trial_types)

@dataclass
class PlayerData:
    """Represents all games for a specific player"""
    pid: str
    games: Dict[str, GameData] = field(default_factory=dict)
    
    def add_game(self, game: GameData) -> None:
        """Add a game to the player's data"""
        self.games[game.game_name] = game
    
    def get_game(self, game_name: str) -> Optional[GameData]:
        """Get data for a specific game"""
        return self.games.get(game_name)
    
    def to_dataframe(self, game_name: Optional[str] = None,
                    min_rt: Optional[float] = None,
                    max_rt: Optional[float] = None) -> pd.DataFrame:
        """
        Convert player's data to DataFrame
        
        Args:
            game_name: Optional game to filter for
            min_rt: Minimum reaction time to include
            max_rt: Maximum reaction time to include
        """
        data = []
        
        games_to_process = [game_name] if game_name else self.games.keys()
        
        for game_name in games_to_process:
            if game_name not in self.games:
                continue
                
            game = self.games[game_name]
            for session in game.sessions.values():
                filtered_trials = session.get_filtered_trials(min_rt=min_rt, max_rt=max_rt)
                for trial in filtered_trials:
                    row = {
                        'PID': self.pid,
                        'Game': game_name,
                        'SessionID': session.session_id,
                        'Timestamp': session.timestamp,
                        'TrialType': trial.trial_type,
                        'ReactionTime': trial.reaction_time,
                        'Success': trial.success
                    }
                    row.update(trial.additional_info)
                    data.append(row)
        
        return pd.DataFrame(data)

class GameDataManager:
    """Manages data for all players"""
    def __init__(self):
        self.players: Dict[str, 'PlayerData'] = {}  # type: ignore

    def _annotate_trials_with_view_location(self, data: Dict[str, Any], trials: List['Trial']) -> None:
        logs = data.get("logs", [])
        if not isinstance(logs, list) or not trials:
            return

        tick_dict = {}
        for entry in logs:
            if entry.get("type") != "HIGH_FREQUENCY_LOG":
                continue

            info = entry.get("info", {})
            tick = info.get("game_tick")
            if tick is None:
                continue

            pitch = info.get("view", {}).get("pitch")
            yaw   = info.get("view", {}).get("yaw")
            loc_x = info.get("location", {}).get("x")
            loc_y = info.get("location", {}).get("y")
            loc_z = info.get("location", {}).get("z")

            block_type = info.get("ray_trace_block", {}).get("block_type")
            block_hit_loc = info.get("ray_trace_block", {}).get("hit_location")  # e.g. {"x":1.5,"y":1.62,"z":39}

            entity = info.get("ray_trace_entities", {}).get("entity")
            entity_hit_loc = info.get("ray_trace_entities", {}).get("hit_location")  # e.g. {"x":1.5,"y":0,"z":41.5}

            # store all 9 pieces (though you might not need them all)
            tick_dict[tick] = (
                pitch, yaw,
                loc_x, loc_y, loc_z,
                block_type, block_hit_loc,
                entity, entity_hit_loc
            )

        # Annotate each trial
        for trial in trials:
            start_tick = trial.start_time
            end_tick   = trial.end_time

            pitch_vals, yaw_vals, x_vals, y_vals, z_vals = [], [], [], [], []
            block_types, block_locs = [], []
            entity_types, entity_locs = [], []

            for t in range(start_tick, end_tick + 1):
                if t in tick_dict:
                    (
                        p, yw,
                        lx, ly, lz,
                        btype, bhit,
                        ent, ehit
                    ) = tick_dict[t]

                    pitch_vals.append(p)
                    yaw_vals.append(yw)
                    x_vals.append(lx)
                    y_vals.append(ly)
                    z_vals.append(lz)
                    block_types.append(btype)
                    block_locs.append(bhit)     # might be None or a dict
                    entity_types.append(ent)
                    entity_locs.append(ehit)
                else:
                    pitch_vals.append(None)
                    yaw_vals.append(None)
                    x_vals.append(None)
                    y_vals.append(None)
                    z_vals.append(None)
                    block_types.append(None)
                    block_locs.append(None)
                    entity_types.append(None)
                    entity_locs.append(None)

            # Store arrays
            trial.additional_info["pitch"]         = pitch_vals
            trial.additional_info["yaw"]           = yaw_vals
            trial.additional_info["pos_x"]         = x_vals
            trial.additional_info["pos_y"]         = y_vals
            trial.additional_info["pos_z"]         = z_vals
            trial.additional_info["block_type"]    = block_types
            trial.additional_info["block_hit_loc"] = block_locs
            trial.additional_info["entity"]        = entity_types
            trial.additional_info["entity_hit_loc"] = entity_locs

            # Then do specialized logic
            if isinstance(trial, DoorDecipherTrial):
                self._compute_door_decipher_metrics(trial)
            elif isinstance(trial, BarnyardBlastTrial):
                self._compute_barnyard_blast_metrics(trial)
            elif isinstance(trial, NetherKnightTrial):
                self._compute_nether_knight_metrics(trial)
            elif isinstance(trial, ParrotSanctuaryTrial):
                self._compute_parrot_sanctuary_metrics(trial)

    def _compute_barnyard_blast_metrics(self, trial: BarnyardBlastTrial) -> None:
        if not isinstance(trial, BarnyardBlastTrial):
            return

        block_types = trial.additional_info.get("block_type", [])
        block_locs  = trial.additional_info.get("block_hit_loc", [])
        # Assume player's position arrays for y and z exist.
        pos_y_arr = trial.additional_info.get("pos_y", [])
        pos_z_arr = trial.additional_info.get("pos_z", [])
        
        start_tick  = trial.start_time
        buffer_ticks = 7
        adjusted_start = start_tick + buffer_ticks

        # 1. Find the LAST target the player saw in the trial.
        last_target = None
        for i in reversed(range(len(block_types))):
            if block_types[i] == "target" and block_locs[i] is not None:
                last_target = block_locs[i]
                break
        if last_target is None:
            trial.additional_info["time_to_target_chosen"] = None
            return

        import math
        # 2. Compute the target center in the y–z plane:
        #    For each coordinate, round down (floor) then add 0.5.
        target_center = {
            "y": math.floor(last_target["y"]) + 0.5,
            "z": math.floor(last_target["z"]) + 0.5,
        }
        radius = 0.125

        # Helper: check if a line segment (from point p to point b) in the y–z plane intersects a circle.
        def segment_intersects_circle(p, b, center, r):
            # p and b are dicts with keys "y" and "z".
            dy = b["y"] - p["y"]
            dz = b["z"] - p["z"]

            # If the segment is effectively a point, check if p is inside the circle.
            if abs(dy) < 1e-6 and abs(dz) < 1e-6:
                dist_sq = (p["y"] - center["y"])**2 + (p["z"] - center["z"])**2
                return dist_sq <= r*r

            # Standard quadratic solution for line-circle intersection.
            # Represent the segment as: P + t*(B-P), with t in [0,1].
            f_y = p["y"] - center["y"]
            f_z = p["z"] - center["z"]

            a = dy*dy + dz*dz
            b_coef = 2 * (f_y * dy + f_z * dz)
            c = f_y*f_y + f_z*f_z - r*r

            discriminant = b_coef*b_coef - 4*a*c
            if discriminant < 0:
                return False  # No intersection.
            discriminant = math.sqrt(discriminant)
            t1 = (-b_coef - discriminant) / (2*a)
            t2 = (-b_coef + discriminant) / (2*a)
            return (0 <= t1 <= 1) or (0 <= t2 <= 1)

        first_fix_tick = None  # We'll store the tick when fixation starts.

        # 3. Iterate over ticks (after buffer) to check for fixation.
        for i in range(len(pos_y_arr)):
            tick_time = start_tick + i
            if tick_time < adjusted_start:
                continue

            # Get player's current position (y and z).
            p_y = pos_y_arr[i]
            p_z = pos_z_arr[i]
            if p_y is None or p_z is None:
                continue
            player_pos = {"y": p_y, "z": p_z}

            # Get the ray_trace_block hit location.
            block_hit = block_locs[i]
            if block_hit is None:
                continue
            b_y = block_hit.get("y")
            b_z = block_hit.get("z")
            if b_y is None or b_z is None:
                continue
            block_pos = {"y": b_y, "z": b_z}

            # Extend the block hit position by 10 blocks in the direction from the player's position.
            dy = block_pos["y"] - player_pos["y"]
            dz = block_pos["z"] - player_pos["z"]
            mag = math.sqrt(dy*dy + dz*dz)
            if mag > 1e-6:
                norm_y = dy / mag
                norm_z = dz / mag
                extended_block_pos = {
                    "y": block_pos["y"] + 10 * norm_y,
                    "z": block_pos["z"] + 10 * norm_z
                }
            else:
                extended_block_pos = block_pos

            # Check if the line segment from player's position to the extended block hit intersects the target circle.
            if segment_intersects_circle(player_pos, extended_block_pos, target_center, radius):
                if first_fix_tick is None:
                    first_fix_tick = tick_time

        if first_fix_tick is None:
            trial.additional_info["time_to_target_chosen"] = None
        else:
            chosen_time = (first_fix_tick - start_tick) / 20.0  # assuming 20 ticks per second.
            trial.additional_info["time_to_target_chosen"] = chosen_time


    def _compute_nether_knight_metrics(self, trial: NetherKnightTrial) -> None:
        """
        This version of the Nether Knight metric:
        1. Finds the location of the last skeleton the player looked at.
        2. Inflates that location by 0.5 on each side (x and z) to define an axis‐aligned hitbox (infinite in y).
        3. For each tick (after a buffer), checks whether the line segment from the player's position to
            their ray_trace_block intersection passes through that hitbox.
        4. Records the first tick (after the buffer) when the player’s gaze remains within the hitbox.
        """
        # Retrieve arrays from trial additional_info.
        entity_types = trial.additional_info.get("entity", [])
        entity_locs  = trial.additional_info.get("entity_hit_loc", [])
        pos_x_arr    = trial.additional_info.get("pos_x", [])
        pos_z_arr    = trial.additional_info.get("pos_z", [])
        block_loc_arr= trial.additional_info.get("block_hit_loc", [])
        start_tick   = trial.start_time
        buffer_ticks = 5
        adjusted_start = start_tick + buffer_ticks

        # 1. Find the last skeleton location.
        last_skeleton = None
        for i in reversed(range(len(entity_types))):
            if entity_types[i] == "skeleton" and entity_locs[i] is not None:
                last_skeleton = entity_locs[i]
                break
        if last_skeleton is None:
            trial.additional_info["time_to_skeleton_chosen"] = None
            return

        # 2. Define the skeleton hitbox (in x-z): inflate 0.5 on either side.
        hitbox = {
            "x_min": last_skeleton["x"] - 0.5,
            "x_max": last_skeleton["x"] + 0.5,
            "z_min": last_skeleton["z"] - 0.5,
            "z_max": last_skeleton["z"] + 0.5,
        }

        # Helper: check if a line segment (from p to b) in the x–z plane intersects the hitbox.
        def segment_intersects_hitbox(p, b, box):
            # p and b are dicts with keys "x" and "z".
            # We use a simple parametric approach (Liang-Barsky algorithm).
            p_x, p_z = p["x"], p["z"]
            r_x = b["x"] - p_x
            r_z = b["z"] - p_z

            # If the ray is essentially a point, check if it lies inside the hitbox.
            if abs(r_x) < 1e-6 and abs(r_z) < 1e-6:
                return (box["x_min"] <= p_x <= box["x_max"] and
                        box["z_min"] <= p_z <= box["z_max"])

            t0, t1 = 0.0, 1.0

            def clip(p_coef, q):
                nonlocal t0, t1
                if abs(p_coef) < 1e-6:
                    # Line is parallel to this boundary: no effect if q >= 0.
                    return q >= 0
                t = q / p_coef
                if p_coef < 0:
                    if t > t1:
                        return False
                    if t > t0:
                        t0 = t
                else:
                    if t < t0:
                        return False
                    if t < t1:
                        t1 = t
                return True

            # For x_min boundary.
            if not clip(-r_x, p_x - box["x_min"]):
                return False
            # For x_max boundary.
            if not clip(r_x, box["x_max"] - p_x):
                return False
            # For z_min boundary.
            if not clip(-r_z, p_z - box["z_min"]):
                return False
            # For z_max boundary.
            if not clip(r_z, box["z_max"] - p_z):
                return False

            # If we have a valid intersection interval, the segment passes through the box.
            return t0 <= t1

        # 3. Now iterate over ticks (after buffer) and check for fixation.
        final_fix_tick = None  # We'll store the tick when fixation starts.
        in_fixation = False
        current_fix_start = None

        for i in range(len(pos_x_arr)):
            tick_time = start_tick + i
            if tick_time < adjusted_start:
                continue

            # Get player's current position (x and z).
            p_x = pos_x_arr[i]
            p_z = pos_z_arr[i]
            if p_x is None or p_z is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            player_pos = {"x": p_x, "z": p_z}

            # Get the ray_trace_block hit location.
            block_hit = block_loc_arr[i]
            if block_hit is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            b_x = block_hit.get("x")
            b_z = block_hit.get("z")
            if b_x is None or b_z is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            block_pos = {"x": b_x, "z": b_z}

            # 4. Check if the line segment from player's pos to block hit goes through the skeleton hitbox.
            if segment_intersects_hitbox(player_pos, block_pos, hitbox):
                if not in_fixation:
                    current_fix_start = tick_time
                    in_fixation = True
            else:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False

        # If the trial ended while still in fixation, use that fixation start.
        if in_fixation and current_fix_start is not None:
            final_fix_tick = current_fix_start

        if final_fix_tick is None:
            trial.additional_info["time_to_skeleton_chosen"] = None
        else:
            # Convert ticks to seconds (assuming 20 ticks per second).
            fixation_time = (final_fix_tick - start_tick) / 20.0
            trial.additional_info["time_to_skeleton_chosen"] = fixation_time

    def _compute_door_decipher_metrics(self, trial: Trial) -> None:
        """
        Compute the time (in seconds) when the player's fixation on the door becomes stable.
        This version:
        1. Uses the player's final position as the center of the door hitbox.
        2. Creates a circular (radial) hitbox with a radius of 0.75 units (infinite in y).
        3. For each tick (after a short buffer), checks if the line segment from the player's current
            position to an extended ray_trace_block hit location (extended 10 blocks forward) 
            intersects the circle.
        4. Records the first tick (after the buffer) when the gaze enters (and remains in) the circle.
        """
        import math

        # Retrieve position arrays and block hit info.
        pos_x_arr = trial.additional_info.get("pos_x", [])
        pos_z_arr = trial.additional_info.get("pos_z", [])
        block_loc_arr = trial.additional_info.get("block_hit_loc", [])
        
        if not pos_x_arr or not pos_z_arr:
            return

        # Get the player's final position.
        final_player_x = pos_x_arr[-1]
        final_player_z = pos_z_arr[-1]
        if final_player_x is None or final_player_z is None:
            trial.additional_info["time_to_door_fixation"] = None
            return

        # Define the center of the door hitbox and set the radius.
        center = {"x": final_player_x, "z": final_player_z}
        radius = 0.75

        # Setup time parameters.
        start_tick = trial.start_time
        buffer_ticks = 5
        adjusted_start = start_tick + buffer_ticks

        # Variables to track fixation.
        final_fix_tick = None      # Tick when fixation starts.
        in_fixation = False        # Whether the gaze is currently within the circle.
        current_fix_start = None   # Start tick of the current fixation period.

        # Helper: check if a line segment (from point p to point b) intersects a circle.
        def segment_intersects_circle(p, b, center, r):
            # p and b are dicts with keys "x" and "z".
            dx = b["x"] - p["x"]
            dz = b["z"] - p["z"]

            # If the segment is essentially a point, simply check if it's inside the circle.
            if abs(dx) < 1e-6 and abs(dz) < 1e-6:
                dist_sq = (p["x"] - center["x"])**2 + (p["z"] - center["z"])**2
                return dist_sq <= r * r

            # Vector from the circle center to p.
            f_x = p["x"] - center["x"]
            f_z = p["z"] - center["z"]

            a = dx*dx + dz*dz
            b_coef = 2 * (f_x * dx + f_z * dz)
            c = f_x*f_x + f_z*f_z - r*r

            discriminant = b_coef*b_coef - 4*a*c
            if discriminant < 0:
                return False
            discriminant = math.sqrt(discriminant)

            t1 = (-b_coef - discriminant) / (2*a)
            t2 = (-b_coef + discriminant) / (2*a)

            return (0 <= t1 <= 1) or (0 <= t2 <= 1)

        # Iterate over ticks (after the buffer) and check for fixation.
        for i in range(len(pos_x_arr)):
            tick_time = start_tick + i
            if tick_time < adjusted_start:
                continue

            # Get the player's current position (x and z).
            p_x = pos_x_arr[i]
            p_z = pos_z_arr[i]
            if p_x is None or p_z is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            player_pos = {"x": p_x, "z": p_z}

            # Get the ray_trace_block hit location.
            block_hit = block_loc_arr[i]
            if block_hit is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            b_x = block_hit.get("x")
            b_z = block_hit.get("z")
            if b_x is None or b_z is None:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False
                continue
            block_pos = {"x": b_x, "z": b_z}

            # Extend the block hit position by 10 blocks in the direction from the player's position.
            dx = block_pos["x"] - player_pos["x"]
            dz = block_pos["z"] - player_pos["z"]
            mag = math.sqrt(dx*dx + dz*dz)
            if mag > 1e-6:
                norm_x = dx / mag
                norm_z = dz / mag
                extended_block_pos = {
                    "x": block_pos["x"] + 10 * norm_x,
                    "z": block_pos["z"] + 10 * norm_z
                }
            else:
                extended_block_pos = block_pos

            # Check if the line segment from player's position to the extended block hit intersects the circle.
            if segment_intersects_circle(player_pos, extended_block_pos, center, radius):
                if not in_fixation:
                    current_fix_start = tick_time
                    in_fixation = True
            else:
                if in_fixation:
                    final_fix_tick = current_fix_start
                    in_fixation = False

        # If the trial ended while still in fixation, record that period.
        if in_fixation and current_fix_start is not None:
            final_fix_tick = current_fix_start

        if final_fix_tick is None:
            trial.additional_info["time_to_door_fixation"] = None
        else:
            fixation_time = (final_fix_tick - start_tick) / 20.0
            trial.additional_info["time_to_door_fixation"] = fixation_time


    def _compute_parrot_sanctuary_metrics(self, trial: ParrotSanctuaryTrial) -> None:
        """
        Analogous to `_compute_barnyard_blast_metrics`, but for Parrot Sanctuary.
        We'll look for the first time 'block_type' == 'target' (or your actual keyword).
        """
        if not isinstance(trial, ParrotSanctuaryTrial):
            return

        block_types = trial.additional_info.get("block_type", [])
        block_locs  = trial.additional_info.get("block_hit_loc", [])
        start_tick  = trial.start_time

        # You can choose buffer_ticks = 7, or something else (like BarnyardBlast).
        buffer_ticks = 7
        distance_threshold = 5  # same approach as Barnyard

        adjusted_start = start_tick + buffer_ticks
        current_loc = None
        first_fix_tick = None

        # Helper: Euclidean distance between two dicts {x:..., y:..., z:...}
        def dist(a, b):
            dx = a["x"] - b["x"]
            dy = a["y"] - b["y"]
            dz = a["z"] - b["z"]
            return (dx*dx + dy*dy + dz*dz)**0.5

        for i, btype in enumerate(block_types):
            t = start_tick + i
            if t < adjusted_start:
                continue  # skip ticks before buffer

            # If your logs say "target" for the relevant parrot or block:
            if btype == "target":
                loc = block_locs[i]  # e.g. {"x":..., "y":..., "z":...} or None
                if loc is None:
                    continue

                if current_loc is None:
                    # First time we fixate on 'target'
                    current_loc = loc
                    first_fix_tick = t
                else:
                    # Compare with the existing current_loc
                    d = dist(loc, current_loc)
                    if d > distance_threshold:
                        # user switched to a new distinct 'target'
                        current_loc = loc
                        first_fix_tick = t

        if first_fix_tick is None:
            trial.time_to_target = None
            trial.additional_info["time_to_target"] = None
        else:
            # Convert ticks to seconds
            chosen_time = (first_fix_tick - start_tick) / 20.0
            trial.time_to_target = chosen_time
            # Also store in additional_info for quick reference
            trial.additional_info["time_to_target"] = chosen_time


    def add_player(self, pid: str) -> 'PlayerData':
        """Add a new player"""
        if pid not in self.players:
            from src.data_structures.game_data import PlayerData  # adjust import path as needed
            self.players[pid] = PlayerData(pid=pid)
        return self.players[pid]
    
    def get_player(self, pid: str) -> Optional['PlayerData']:
        """Get player data"""
        return self.players.get(pid)

    def _process_build_master(self, data: Dict, info: Dict) -> List['Trial']:
        """Process BUILD_MASTER game data into trials"""
        from src.data_structures.game_data import Trial  # adjust import path as needed
        
        trials = []
        level_completions = {}  # Track which levels were completed
        
        # Get game record
        game_result = data.get('game_result', {})
        game_record = game_result.get('game_record', {})
        
        # Skip old format
        if isinstance(game_record, str):
            return trials
        
        game_mode = game_record.get('game_mode', 'UNKNOWN')
        level_records = game_record.get('level_records', {})
        
        # First pass: identify completed levels
        if isinstance(level_records, dict):
            for record in level_records.values():
                if isinstance(record, dict):
                    difficulty = record.get('difficulty')
                    if difficulty is not None and record.get('passed', False):
                        level_completions[difficulty] = True
        
        # Process task summary for attempts
        task_summary = data.get('task_item_summary', {})
        if isinstance(task_summary, dict):
            task_summary = task_summary.values()
            
        for attempt in task_summary:
            for state in attempt.values():
                if isinstance(state, dict) and state.get('state') == 'BUILD_STATE':
                    info_state = state.get('info', {})
                    difficulty = info_state.get('difficulty', 0)
                    
                    start_tick = state.get('state_start_game_tick', 0)
                    end_tick = state.get('state_end_game_tick', 0)
                    
                    # Create trial
                    trial = Trial(
                        start_time=start_tick,
                        end_time=end_tick,
                        duration=end_tick - start_tick,
                        success=(difficulty in level_completions),
                        trial_type=game_mode,
                        additional_info={
                            'difficulty': difficulty,
                            'game_mode': game_mode,
                            'session_number': info_state.get('session_number', 1)
                        }
                    )
                    trials.append(trial)
        
        return trials

    def _process_slime_squash(self, data: Dict, info: Dict) -> List['Trial']:
        """Process SLIME_SQUASH game data into trials"""
        from src.data_structures.game_data import Trial  # adjust import path as needed
        
        trials = []
        
        task_summary = data.get('task_item_summary', {})
        game_result = data.get('game_result', {})
        total_errors = game_result.get('total_errors', 0)
        if isinstance(task_summary, dict):
            task_summary = task_summary.values()
        
        total_attempts = 0
        for attempt in task_summary:
            state = attempt.get('0', {})
            if state.get('state') == 'ACTION_STATE':
                start_tick = state.get('state_start_game_tick', 0)
                end_tick = state.get('state_end_game_tick', 0)
                
                # Get the info dictionary once
                info = state.get('info', {})
                
                # Extract each field explicitly, providing defaults if necessary
                additional_info = {
                    'slime_spawn': info.get('slime_spawn', {}),
                    'slime_hits': info.get('slime_hits', {}),
                    'task_item_product': info.get('task_item_product', False),
                    'time_elapsed': info.get('time_elapsed', 0),
                    'slime_count': info.get('slime_count', 0),
                    'task_type': info.get('task_type', 'unknown')
                }
                # Determine success based on the specific field in info
                success = info.get('task_item_product', False)
                
                trial = Trial(
                    start_time=start_tick,
                    end_time=end_tick,
                    duration=end_tick - start_tick,
                    success=success,
                    trial_type='standard',
                    additional_info=additional_info
                )
                trials.append(trial)
                total_attempts += 1
        return trials

    def _process_barnyard_blast(self, data: Dict, info: Dict) -> List[BarnyardBlastTrial]:
        """
        Process BARNYARD_BLAST data into specialized BarnyardBlastTrial objects,
        calling _extract_additional_info() to parse out 'animal', 'direction', etc.
        """
        trials = []
        task_summary = data.get('task_item_summary', {})

        if isinstance(task_summary, dict):
            task_summary = list(task_summary.values())

        for attempt in task_summary:
            # Typically, each attempt is a dict like {'0': {...}, '1': {...}, ...}
            state = attempt.get('0', {})
            if state.get('state') != 'ACTION_STATE':
                continue

            info_data = state.get('info', {})
            start_tick = state.get('state_start_game_tick', 0)
            end_tick   = state.get('state_end_game_tick', 0)
            success    = info_data.get('task_item_product', False)

            # Here is where we actually call the parser:
            parsed_info = self._extract_additional_info('BARNYARD_BLAST', info_data)

            trial = BarnyardBlastTrial(
                start_time=start_tick,
                end_time=end_tick,
                duration=end_tick - start_tick,
                success=success,
                trial_type="BARNYARD_BLAST",  # so we know this is a Barnyard Blast trial
                additional_info={
                    "info": info_data,
                    **parsed_info  # merges 'animal', 'direction', 'congruency', etc.
                }
            )

            trials.append(trial)

        return trials

    def _process_nether_knight(self, data: Dict, info: Dict) -> List[NetherKnightTrial]:
        """
        Process NETHER_KNIGHT data into specialized NetherKnightTrial objects, 
        which will later be annotated with time_to_skeleton.
        """
        trials = []
        task_summary = data.get('task_item_summary', {})

        if isinstance(task_summary, dict):
            task_summary = list(task_summary.values())

        for attempt in task_summary:
            state = attempt.get('0', {})
            if state.get('state') != 'ACTION_STATE':
                continue
            
            info_data = state.get('info', {})
            start_tick = state.get('state_start_game_tick', 0)
            end_tick   = state.get('state_end_game_tick', 0)
            success    = info_data.get('task_item_product', False)
            
            # Create a NetherKnightTrial
            trial = NetherKnightTrial(
                start_time=start_tick,
                end_time=end_tick,
                duration=end_tick - start_tick,
                success=success,
                trial_type="NETHER_KNIGHT",
                additional_info={
                    "info": info_data,
                }
            )
            trials.append(trial)

        return trials

    def _process_door_decipher(self, data: Dict, info: Dict) -> List[DoorDecipherTrial]:
        from src.data_structures.game_data import DoorDecipherTrial

        trials = []
        task_summary = data.get('task_item_summary', {})
        if isinstance(task_summary, dict):
            task_summary = list(task_summary.values())

        # Initialize state tracking variables
        previous_rule = None
        is_searching_flag = True

        for attempt in task_summary:
            state = attempt.get('0', {})
            if state.get('state') != 'ACTION_STATE':
                continue

            info_data = state.get('info', {})
            current_rule = info_data.get('current_rule')

            # If this is the first trial or the rule has changed, reset is_searching
            if previous_rule is None or current_rule != previous_rule:
                is_searching_flag = True

            # Pass the computed flag into your additional info extractor.
            # (This helper function will store the current_rule and is_searching flag.)
            parsed_info = self._extract_additional_info('DOOR_DECIPHER', info_data, is_searching=is_searching_flag)
            
            start_tick = state.get('state_start_game_tick', 0)
            end_tick = state.get('state_end_game_tick', 0)
            success = info_data.get('task_item_product', True)

            trial = DoorDecipherTrial(
                start_time=start_tick,
                end_time=end_tick,
                duration=end_tick - start_tick,
                success=success,
                trial_type="DOOR_DECIPHER",
                additional_info={
                    "info": info_data,
                    **parsed_info
                }
            )
            trials.append(trial)

            # If task_item_product is true, then (after this trial) the search should stop.
            if success:
                is_searching_flag = False

            # Update the rule for the next iteration
            previous_rule = current_rule

        return trials


    def _process_parrot_sanctuary(self, data: Dict[str, Any]) -> List[ParrotSanctuaryTrial]:
        trials = []
        task_summary = data.get('task_item_summary', {})
        if isinstance(task_summary, dict):
            task_summary = list(task_summary.values())

        for attempt in task_summary:
            state = attempt.get('0', {})
            if state.get('state') != 'ACTION_STATE':
                continue

            info_data = state.get('info', {})
            start_tick = state.get('state_start_game_tick', 0)
            end_tick   = state.get('state_end_game_tick', 0)
            success    = info_data.get('task_item_product', False)

            trial = ParrotSanctuaryTrial(
                start_time=start_tick,
                end_time=end_tick,
                duration=end_tick - start_tick,
                success=success,
                trial_type="PARROT_SANCTUARY",
                additional_info={
                    "info": info_data
                }
            )
            trials.append(trial)
        return trials

    def load_from_json(self, file_path: Union[str, Path], username_filter: str = 'nmil') -> None:
        """Load data from JSON file with special game processing and include 'finished' flag."""
        from src.data_structures.game_data import GameData, GameSession, Trial  # adjust import path as needed
        
        file_path = Path(file_path)
        pid = file_path.parent.name
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        username = data.get('username', '')
        if username_filter.lower() not in username.lower():
            return
        
        player = self.add_player(pid)
        
        minigame = data.get('minigame', '')
        if not minigame:
            return
        
        # Strip version info from minigame name (e.g. "NETHER_KNIGHT_V0_5_0" -> "NETHER_KNIGHT")
        base_minigame = re.sub(r'_V\d+_\d+_\d+$', '', minigame)
        
        # Get/create GameData object for this minigame
        if base_minigame not in player.games:
            player.games[base_minigame] = GameData(game_name=base_minigame)
        game_data = player.games[base_minigame]
        
        # Create a new session
        timestamp_str = file_path.stem
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            timestamp = datetime.now()
        
        session = GameSession(session_id=timestamp_str, timestamp=timestamp)
        
        # Include 'finished' if present
        session.metadata['finished'] = data.get('finished', False)
        
        # Process trials based on the minigame
        if base_minigame == 'BUILD_MASTER':
            trials = self._process_build_master(data, data.get('info', {}))
            # Annotate with position and view data
            self._annotate_trials_with_view_location(data, trials)

            if game_data.build_master_data is None:
                from src.data_structures.game_data import BuildMasterData
                game_data.build_master_data = BuildMasterData()
            
            for trial in trials:
                game_data.build_master_data.add_attempt(trial)
        
        elif base_minigame == 'SLIME_SQUASH':
            trials = self._process_slime_squash(data, data.get('info', {}))
            # Annotate with position and view data
            self._annotate_trials_with_view_location(data, trials)

        elif base_minigame == 'BARNYARD_BLAST':
            # <-- Barnyard Blast specialized method
            trials = self._process_barnyard_blast(data, data.get('info', {}))
            self._annotate_trials_with_view_location(data, trials)

        elif base_minigame == 'NETHER_KNIGHT':
            trials = self._process_nether_knight(data, data.get('info', {}))
            self._annotate_trials_with_view_location(data, trials)

        elif base_minigame == 'DOOR_DECIPHER':
            trials = self._process_door_decipher(data, data.get('info', {}))
            self._annotate_trials_with_view_location(data, trials)
        elif base_minigame == 'PARROT_SANCTUARY':
            trials = self._process_parrot_sanctuary(data)
            self._annotate_trials_with_view_location(data, trials)

        else:
            # Some other minigame
            trials = []
            task_summary = data.get('task_item_summary', {})
            if isinstance(task_summary, dict):
                task_summary = list(task_summary.values())
            
            previous_rule = None
            is_searching = True
            
            for attempt in task_summary:
                state = attempt.get('0', {})
                if state.get('state') != 'ACTION_STATE':
                    continue

                info_data = state.get('info', {})
                
                # Handle DOOR_DECIPHER logic
                if base_minigame == 'DOOR_DECIPHER':
                    current_rule = info_data.get('current_rule')
                    if previous_rule is not None and current_rule != previous_rule:
                        is_searching = True
                    elif is_searching and info_data.get('task_item_product', False):
                        is_searching = False
                    previous_rule = current_rule
                                
                trial = Trial(
                    start_time=state.get('state_start_game_tick', 0),
                    end_time=state.get('state_end_game_tick', 0),
                    duration=state.get('state_end_game_tick', 0) - state.get('state_start_game_tick', 0),
                    success=info_data.get('task_item_product', False),
                    trial_type=self._determine_trial_type(base_minigame, info_data, is_searching),
                    additional_info={
                        "info": info_data
                    }
                )
                trials.append(trial)

            # Annotate with position and view data
            self._annotate_trials_with_view_location(data, trials)
        
        # Add all trials to the session
        for trial in trials:
            session.add_trial(trial)
        
        # If there are trials, store this session
        if session.trials:
            game_data.add_session(session)

    def _determine_trial_type(self, minigame: str, info: Dict, is_searching: bool = False) -> str:
        """Determine trial type based on minigame and info"""
        if minigame == 'DOOR_DECIPHER':
            if is_searching:
                return 'discovery'
            return 'flow'
        elif minigame == 'BARNYARD_BLAST':
            return info.get('animal_pattern', 'unknown')
        elif minigame == 'PARROT_SANCTUARY':
            return info.get('parrot_spawn', 'unknown')
        return 'standard'
    
    def _extract_additional_info(self, minigame: str, info: Dict, is_searching: bool = False) -> Dict:
        """Extract additional info specific to each minigame"""
        additional_info = {}
        
        if minigame == 'DOOR_DECIPHER':
            current_rule = info.get('current_rule')
            additional_info.update({
                'current_rule': current_rule,
                'is_searching': is_searching
            })
        elif minigame == 'BARNYARD_BLAST':
            pattern = info.get('animal_pattern', '')
            parts = re.findall('[A-Z][^A-Z]*', pattern)
            additional_info.update({
                'congruency': parts[0] if len(parts) > 0 else '',
                'direction': parts[1] if len(parts) > 1 else '',
                'animal': parts[2] if len(parts) > 2 else ''
            })
        elif minigame == 'PARROT_SANCTUARY':
            spawn = info.get('parrot_spawn', '')
            parts = re.findall('[A-Z][^A-Z]*', spawn)
            additional_info.update({
                'color': parts[0] if len(parts) > 0 else '',
                'position': parts[1] if len(parts) > 1 else ''
            })
        
        return additional_info
    
    def to_dataframe(self, game_name: Optional[str] = None,
                     min_rt: Optional[float] = None,
                     max_rt: Optional[float] = None) -> 'pd.DataFrame':
        """Convert all data to a pandas DataFrame."""
        import pandas as pd
        dfs = []
        
        for player in self.players.values():
            df = player.to_dataframe(game_name, min_rt=min_rt, max_rt=max_rt)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    def save_to_pickle(self, file_path: Union[str, Path]) -> None:
        """Save manager state to pickle file"""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_pickle(cls, file_path: Union[str, Path]) -> 'GameDataManager':
        """Load manager state from pickle file"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
def compute_lswm_highest_forms(trials: List[Trial]) -> Tuple[float, float]:
    """
    Given a list of Trial objects from List Sorting Working Memory (LSWM),
    compute the highest form completed for 1-list items and 2-list items.

    Returns:
        (highest_form_1, highest_form_2) as floats (or np.nan if none).
    """

    # Maps the final character of ItemID to the sequence length
    sequence_length_map = {
        'A': 2, 'B': 2, 'M': 2, 'N': 2,
        'C': 3, 'D': 3, 'O': 3, 'P': 3,
        'E': 4, 'F': 4, 'Q': 4, 'R': 4,
        'G': 5, 'H': 5, 'S': 5, 'T': 5,
        'I': 6, 'J': 6, 'U': 6, 'V': 6,
        'K': 7, 'L': 7, 'W': 7, 'X': 7
    }

    # Filter only LSWM trials
    # (Optional: check item ID starts with "LSWM_", or rely on game name check)
    lswm_trials = [
        t for t in trials 
        if t.additional_info.get("ItemID", "").startswith("LSWM_") 
    ]
    if not lswm_trials:
        return (float('nan'), float('nan'))

    # Helper to get highest form for 1-list or 2-list
    def get_highest_for_list(list_number: int) -> float:
        # e.g., "LSWM_1List_Live_Item_"
        prefix = f"LSWM_{list_number}List_Live_Item_"
        # Find all successful trials with matching prefix
        completed = [
            t for t in lswm_trials
            if t.success 
            and prefix in t.additional_info.get("ItemID", "")
        ]
        if not completed:
            return float('nan')
        
        # Extract the last character from each ItemID
        last_chars = [trial.additional_info["ItemID"][-1] for trial in completed]
        # Map those chars to sequence lengths
        mapped_lengths = [sequence_length_map.get(ch, float('nan')) for ch in last_chars]
        # Return the max
        return max(mapped_lengths) if mapped_lengths else float('nan')

    highest_form_1 = get_highest_for_list(1)
    highest_form_2 = get_highest_for_list(2)
    return (highest_form_1, highest_form_2)