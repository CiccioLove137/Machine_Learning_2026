from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


WINDOWS_DIR = Path("data/processed/windows")
SPLITS_DIR = Path("data/processed/splits")


def _load_subject_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray]:
    """
    Ritorna:
      X: (N,T,F)
      y: (N,)
      subjects: (N,) stringhe ripetute (subjectXXX)
      feature_cols: (F,)
      window_size: int
      stride: int
      activities: (C,)
    """
    data = np.load(path, allow_pickle=True)

    if "X" not in data or "y" not in data:
        raise ValueError(f"File {path} non contiene X/y.")

    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    # Nei per-soggetto noi avevamo salvato "subject" singolo.
    # Qui lo espandiamo a un array lungo N.
    if "subject" in data:
        subj = str(data["subject"][0])
        subjects = np.array([subj] * X.shape[0], dtype=object)
    elif "subjects" in data:
        subjects = data["subjects"].astype(object)
    else:
        # fallback: usa nome file
        subj = path.stem
        subjects = np.array([subj] * X.shape[0], dtype=object)

    feature_cols = data["feature_cols"].astype(object) if "feature_cols" in data else np.array([], dtype=object)
    window_size = int(data["window_size"]) if "window_size" in data else -1
    stride = int(data["stride"]) if "stride" in data else -1
    activities = data["activities"].astype(np.int32) if "activities" in data else np.array([], dtype=np.int32)

    return X, y, subjects, feature_cols, window_size, stride, activities


def _concat_parts(parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    parts: lista di (X, y, subjects)
    """
    if not parts:
        return (
            np.empty((0, 0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=object),
        )

    Xs, ys, ss = zip(*parts)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    subjects = np.concatenate(ss, axis=0)
    return X, y, subjects


def _check_unique_sets(train: List[str], val: List[str], test: List[str]) -> None:
    all_sets = {"train": set(train), "val": set(val), "test": set(test)}
    inter_tv = all_sets["train"] & all_sets["val"]
    inter_tt = all_sets["train"] & all_sets["test"]
    inter_vt = all_sets["val"] & all_sets["test"]
    if inter_tv or inter_tt or inter_vt:
        raise ValueError(
            f"Soggetti in più set!\n"
            f"train∩val={sorted(inter_tv)}\n"
            f"train∩test={sorted(inter_tt)}\n"
            f"val∩test={sorted(inter_vt)}"
        )


def _save_split_npz(
    out_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    meta: Dict,
) -> None:
    np.savez_compressed(
        out_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        subjects=subjects.astype(object),
        feature_cols=np.array(meta["feature_cols"], dtype=object),
        activities=np.array(meta["activities"], dtype=np.int32),
        window_size=np.int32(meta["window_size"]),
        stride=np.int32(meta["stride"]),
        sample_rate=np.int32(meta["sample_rate"]),
        window_sec=np.int32(meta["window_sec"]),
        overlap=np.float32(meta["overlap"]),
        purity_threshold=np.float32(meta["purity_threshold"]),
        use_hr=np.bool_(meta["use_hr"]),
        hr_fill_strategy=np.array([meta["hr_fill_strategy"]], dtype=object),
        remove_orientation=np.bool_(meta["remove_orientation"]),
        split_name=np.array([meta["split_name"]], dtype=object),
        split_subjects=np.array(meta["split_subjects"], dtype=object),
        seed=np.int32(meta["seed"]),
    )


def run_split(cfg: dict) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Leggi split da YAML
    split_cfg = cfg["split"]
    train_only = list(split_cfg.get("train_only_subjects", []))
    train_subs = list(split_cfg.get("train_subjects", [])) + train_only
    val_subs = list(split_cfg.get("val_subjects", []))
    test_subs = list(split_cfg.get("test_subjects", []))

    _check_unique_sets(train_subs, val_subs, test_subs)

    # Metadati da YAML 
    sample_rate = int(cfg["dataset"]["sample_rate"])
    window_sec = int(cfg["windowing"]["window_sec"])
    overlap = float(cfg["windowing"]["overlap"])
    purity_threshold = float(cfg["windowing"]["purity_threshold"])
    use_hr = bool(cfg["dataset"].get("use_hr", True))
    hr_fill_strategy = str(cfg["dataset"].get("hr_fill_strategy", "simple"))
    remove_orientation = bool(cfg["dataset"].get("remove_orientation", True))
    seed = int(cfg.get("seed", 42))

    def load_many(subjects: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        ref_feature_cols = None
        ref_window_size = None
        ref_stride = None
        ref_activities = None

        for s in subjects:
            p = WINDOWS_DIR / f"{s}.npz"
            if not p.exists():
                raise FileNotFoundError(f"Non trovo {p}. Hai eseguito stage windows?")

            X, y, subj_arr, feature_cols, window_size, stride, activities = _load_subject_npz(p)

            # Controlli consistenza feature/parametri
            if ref_feature_cols is None:
                ref_feature_cols = feature_cols
                ref_window_size = window_size
                ref_stride = stride
                ref_activities = activities
            else:
                if len(feature_cols) != len(ref_feature_cols) or not np.array_equal(feature_cols, ref_feature_cols):
                    raise ValueError(f"Feature_cols non consistenti tra soggetti. Problema su {s}.")
                if window_size != ref_window_size or stride != ref_stride:
                    raise ValueError(f"window_size/stride non consistenti. Problema su {s}.")
                if not np.array_equal(activities, ref_activities):
                    raise ValueError(f"Lista activities non consistente. Problema su {s}.")

            parts.append((X, y, subj_arr))

        X_all, y_all, subjects_all = _concat_parts(parts)

        meta = {
            "feature_cols": ref_feature_cols if ref_feature_cols is not None else np.array([], dtype=object),
            "window_size": ref_window_size if ref_window_size is not None else -1,
            "stride": ref_stride if ref_stride is not None else -1,
            "activities": ref_activities if ref_activities is not None else np.array([], dtype=np.int32),
        }
        return X_all, y_all, subjects_all, meta

    print(f"WINDOWS_DIR: {WINDOWS_DIR}")
    print(f"SPLITS_DIR:  {SPLITS_DIR}\n")
    print(f"Train subjects: {train_subs}")
    print(f"Val subjects:   {val_subs}")
    print(f"Test subjects:  {test_subs}\n")

    # Carica e concatena
    X_train, y_train, s_train, meta_train = load_many(train_subs)
    X_val, y_val, s_val, meta_val = load_many(val_subs)
    X_test, y_test, s_test, meta_test = load_many(test_subs)

    # Meta comune (prendiamo dal train come riferimento)
    common_meta = {
        "feature_cols": meta_train["feature_cols"],
        "window_size": meta_train["window_size"],
        "stride": meta_train["stride"],
        "activities": meta_train["activities"],
        "sample_rate": sample_rate,
        "window_sec": window_sec,
        "overlap": overlap,
        "purity_threshold": purity_threshold,
        "use_hr": use_hr,
        "hr_fill_strategy": hr_fill_strategy,
        "remove_orientation": remove_orientation,
        "seed": seed,
    }

    # Salva split completi
    _save_split_npz(
        SPLITS_DIR / "train.npz",
        X_train,
        y_train,
        s_train,
        meta={**common_meta, "split_name": "train", "split_subjects": np.array(train_subs, dtype=object)},
    )
    _save_split_npz(
        SPLITS_DIR / "val.npz",
        X_val,
        y_val,
        s_val,
        meta={**common_meta, "split_name": "val", "split_subjects": np.array(val_subs, dtype=object)},
    )
    _save_split_npz(
        SPLITS_DIR / "test.npz",
        X_test,
        y_test,
        s_test,
        meta={**common_meta, "split_name": "test", "split_subjects": np.array(test_subs, dtype=object)},
    )

    print("Salvati:")
    print(f" - {SPLITS_DIR / 'train.npz'} | X={X_train.shape} y={y_train.shape}")
    print(f" - {SPLITS_DIR / 'val.npz'}   | X={X_val.shape} y={y_val.shape}")
    print(f" - {SPLITS_DIR / 'test.npz'}  | X={X_test.shape} y={y_test.shape}")
