from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# PAMAP2: 54 colonne
N_COLS = 54
COL_TIMESTAMP = 0
COL_ACTIVITY = 1
COL_HR = 2

HAND_COLS = list(range(3, 20))
CHEST_COLS = list(range(20, 37))
ANKLE_COLS = list(range(37, 54))


def _imu_orientation_cols(block_start_0idx: int) -> List[int]:
    # quaternion/orientation: 4 colonne nel PAMAP2
    return [block_start_0idx + off for off in (13, 14, 15, 16)]


ORIENT_COLS = (
    _imu_orientation_cols(HAND_COLS[0])
    + _imu_orientation_cols(CHEST_COLS[0])
    + _imu_orientation_cols(ANKLE_COLS[0])
)


def list_subject_files(protocol_dir: Path) -> Dict[str, Path]:
    files = sorted(protocol_dir.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"Nessun file .dat trovato in: {protocol_dir}")
    return {fp.stem: fp for fp in files}


def load_dat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != N_COLS:
        print(f"[WARN] {path.name}: colonne={df.shape[1]} (attese {N_COLS})")

    # rinomina colonne principali
    df = df.rename(columns={COL_TIMESTAMP: "Timestamp", COL_ACTIVITY: "ActivityID", COL_HR: "HR"})
    return df


def apply_filters(df: pd.DataFrame, activities: List[int], exclude_activities: List[int]) -> pd.DataFrame:
    out = df.copy()
    if exclude_activities:
        out = out[~out["ActivityID"].isin(exclude_activities)]
    out = out[out["ActivityID"].isin(activities)]
    return out.reset_index(drop=True)


def fill_hr_simple(df: pd.DataFrame) -> pd.DataFrame:
    # scelta: ffill + bfill
    out = df.copy()
    out["HR"] = out["HR"].ffill().bfill()
    return out


def remove_orientation_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in ORIENT_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop) if cols_to_drop else df


def fill_imu_nans_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    Versione base:
    - IMU: interpolazione lineare + ffill/bfill
    - se resta NaN: fill 0
    """
    out = df.copy()
    feature_cols = [c for c in out.columns if c not in ["Timestamp", "ActivityID", "HR"]]
    if not feature_cols:
        return out

    out[feature_cols] = out[feature_cols].astype("float64").interpolate(method="linear", limit_direction="both")
    out[feature_cols] = out[feature_cols].ffill().bfill()
    if out[feature_cols].isna().any().any():
        out[feature_cols] = out[feature_cols].fillna(0.0)

    return out


def make_windows(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    act_to_idx: Dict[int, int],
    purity_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """
    Crea finestre (N, T, F) e label (N,).
    Label finestra = majority vote sui T campioni.
    Scarta finestre con purezza < purity_threshold.
    """
    feature_cols = [c for c in df.columns if c not in ["Timestamp", "ActivityID"]]
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["ActivityID"].to_numpy(dtype=np.int32)

    n = len(df)
    stats = {"kept": 0, "skipped_impure": 0, "skipped_bad": 0}

    if n < window_size:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_cols,
            stats,
        )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        Xw = X_all[start:end, :]

        if not np.isfinite(Xw).all():
            stats["skipped_bad"] += 1
            continue

        y_slice = y_all[start:end]
        vals, counts = np.unique(y_slice, return_counts=True)

        maj_i = int(np.argmax(counts))
        maj_act = int(vals[maj_i])

        purity = float(counts[maj_i]) / float(window_size)
        if purity < purity_threshold:
            stats["skipped_impure"] += 1
            continue

        if maj_act not in act_to_idx:
            stats["skipped_bad"] += 1
            continue

        X_list.append(Xw)
        y_list.append(act_to_idx[maj_act])
        stats["kept"] += 1

    if not X_list:
        return (
            np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            feature_cols,
            stats,
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, feature_cols, stats


def run_make_windows(cfg: dict) -> None:
    raw_dir = Path(cfg["dataset"]["raw_protocol_dir"])
    out_dir = Path("data/processed/windows")
    out_dir.mkdir(parents=True, exist_ok=True)

    activities = [int(a) for a in cfg["dataset"]["activities"]]
    exclude_acts = [int(a) for a in cfg["dataset"].get("exclude_activities", [])]
    exclude_subjects = set(cfg["dataset"].get("exclude_subjects", []))

    sample_rate = int(cfg["dataset"]["sample_rate"])
    window_sec = int(cfg["windowing"]["window_sec"])
    overlap = float(cfg["windowing"]["overlap"])
    purity_threshold = float(cfg["windowing"]["purity_threshold"])

    window_size = window_sec * sample_rate
    stride = int(window_size * (1.0 - overlap))

    use_hr = bool(cfg["dataset"].get("use_hr", True))
    hr_strategy = str(cfg["dataset"].get("hr_fill_strategy", "simple"))
    remove_orient = bool(cfg["dataset"].get("remove_orientation", True))

    if use_hr and hr_strategy != "simple":
        raise ValueError("In questa versione supportiamo solo hr_fill_strategy: simple (ffill+bfill).")

    act_to_idx = {a: i for i, a in enumerate(activities)}

    subject_files = list_subject_files(raw_dir)
    for s in list(exclude_subjects):
        subject_files.pop(s, None)

    print(f"RAW: {raw_dir}")
    print(f"OUT: {out_dir}")
    print(f"Soggetti: {list(subject_files.keys())}")
    print(f"window_size={window_size} stride={stride} purity_threshold={purity_threshold}")
    print(f"use_hr={use_hr} hr_strategy={hr_strategy} remove_orientation={remove_orient}\n")

    for subject, fp in subject_files.items():
        df_raw = load_dat(fp)
        df = apply_filters(df_raw, activities, exclude_acts)

        if use_hr:
            df = fill_hr_simple(df)
        else:
            if "HR" in df.columns:
                df = df.drop(columns=["HR"])

        if remove_orient:
            df = remove_orientation_columns(df)

        df = fill_imu_nans_simple(df)

        X, y, feature_cols, stats = make_windows(df, window_size, stride, act_to_idx, purity_threshold)

        print(
            f"[{subject}] rows={len(df):>7} windows={X.shape[0]:>6} | "
            f"kept={stats['kept']} impure={stats['skipped_impure']} bad={stats['skipped_bad']}"
        )

        np.savez_compressed(
            out_dir / f"{subject}.npz",
            X=X,
            y=y,
            subject=np.array([subject], dtype=object),
            feature_cols=np.array(feature_cols, dtype=object),
            window_size=np.int32(window_size),
            stride=np.int32(stride),
            activities=np.array(activities, dtype=np.int32),
        )

    print("\nDone. Salvati file per-soggetto in data/processed/windows/")
