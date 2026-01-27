# scripts/00_dataset_diagnosis_pamap2.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG MINIMA (standalone)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PROTOCOL_DIR = PROJECT_ROOT / "data" / "raw" / "Protocol"
OUT_DIR = PROJECT_ROOT / "reports" / "diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PAMAP2 Protocol .dat: 54 colonne, con:
# 0 Timestamp, 1 ActivityID, 2 HR, poi sensori
N_COLS = 54
COL_TIMESTAMP = 0
COL_ACTIVITY = 1
COL_HR = 2


# Teniamo solo queste activity
ACTIVITIES = [1,2,3,4,5,6,7,12,13,16,17]  # oppure None per non filtrare
EXCLUDE_ACTIVITIES = {0, 24}              # attività da escludere sempre

# Per escludere soggetti problematici:
EXCLUDE_SUBJECTS = {"subject109"}         # oppure set()

# Se vogliamo includere HR nel report missingness (utile per capire quanto manca)
CHECK_HR = True


# =========================
# LOAD
# =========================
def list_subject_files(protocol_dir: Path) -> Dict[str, Path]:
    files = sorted(protocol_dir.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"Nessun .dat trovato in: {protocol_dir}")
    return {fp.stem: fp for fp in files}

def load_dat(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != N_COLS:
        print(f"[WARN] {file_path.name}: colonne={df.shape[1]} (attese {N_COLS})")
    df = df.rename(columns={COL_TIMESTAMP: "Timestamp", COL_ACTIVITY: "ActivityID", COL_HR: "HR"})
    return df

def apply_basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if EXCLUDE_ACTIVITIES:
        out = out[~out["ActivityID"].isin(EXCLUDE_ACTIVITIES)]
    if ACTIVITIES is not None:
        out = out[out["ActivityID"].isin(ACTIVITIES)]
    return out.reset_index(drop=True)

def compute_missingness(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Ritorna:
      - n_nan per colonna
      - n_inf per colonna (solo per colonne numeriche)
    """
    n_nan = df.isna().sum()

    # Inf solo su numeriche
    num_df = df.select_dtypes(include=[np.number])
    n_inf = pd.Series(0, index=df.columns)
    if not num_df.empty:
        inf_mask = np.isinf(num_df.to_numpy())
        n_inf.loc[num_df.columns] = inf_mask.sum(axis=0)

    return n_nan, n_inf

def activity_counts(df: pd.DataFrame) -> pd.Series:
    return df["ActivityID"].value_counts().sort_index()

def ensure_all_activities_indexed(counts: pd.Series) -> pd.Series:
    """
    Se ACTIVITIES è definito, forza l'indice a contenere tutte le activity target.
    """
    if ACTIVITIES is None:
        return counts
    idx = pd.Index(ACTIVITIES, dtype=int)
    return counts.reindex(idx, fill_value=0)


# =========================
# PLOTS 
# =========================
def plot_subject_activity_bar(subject: str, counts: pd.Series, out_dir: Path) -> None:
    plt.figure()
    plt.bar([str(i) for i in counts.index.tolist()], counts.values.tolist())
    plt.title(f"{subject} - ActivityID counts (rows)")
    plt.xlabel("ActivityID")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / f"{subject}_activity_counts.png", dpi=200)
    plt.close()

def plot_missingness_bar(subject: str, n_nan: pd.Series, n_inf: pd.Series, out_dir: Path) -> None:
    # Tieni solo colonne con qualche problema
    bad = (n_nan > 0) | (n_inf > 0)
    n_nan_bad = n_nan[bad]
    n_inf_bad = n_inf[bad]

    # Se non ci sono problemi, salva comunque un file "vuoto" informativo
    if n_nan_bad.empty and n_inf_bad.empty:
        plt.figure()
        plt.title(f"{subject} - No NaN/Inf detected")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{subject}_missingness.png", dpi=200)
        plt.close()
        return

    plt.figure(figsize=(10, 4))
    x = np.arange(len(n_nan_bad.index))
    plt.bar(x - 0.2, n_nan_bad.values, width=0.4, label="NaN")
    plt.bar(x + 0.2, n_inf_bad.values, width=0.4, label="Inf")
    plt.title(f"{subject} - Missingness per column (only problematic cols)")
    plt.xlabel("Column")
    plt.ylabel("Count")
    plt.xticks(x, [str(c) for c in n_nan_bad.index], rotation=90, fontsize=7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{subject}_missingness.png", dpi=200)
    plt.close()

def plot_activity_matrix_heatmap(subjects: List[str], matrix: np.ndarray, activities: List[int], out_dir: Path) -> None:
    """
    Heatmap soggetti per attività (conteggi righe).
    """
    plt.figure(figsize=(10, max(3, 0.35 * len(subjects))))
    plt.imshow(matrix, aspect="auto")
    plt.title("Subjects x ActivityID (row counts)")
    plt.xlabel("ActivityID")
    plt.ylabel("Subject")
    plt.xticks(np.arange(len(activities)), [str(a) for a in activities], rotation=45)
    plt.yticks(np.arange(len(subjects)), subjects)
    plt.colorbar(label="Count")
    plt.tight_layout()
    plt.savefig(out_dir / "ALL_subjects_activity_heatmap.png", dpi=220)
    plt.close()

def plot_missing_cols_matrix(subjects: List[str], matrix: np.ndarray, cols: List[str], out_dir: Path) -> None:
    """
    Heatmap soggetti x colonne (NaN count). Mostra solo le colonne più problematiche.
    """
    plt.figure(figsize=(12, max(3, 0.35 * len(subjects))))
    plt.imshow(matrix, aspect="auto")
    plt.title("Subjects x Columns (NaN count) - top problematic columns")
    plt.xlabel("Column")
    plt.ylabel("Subject")
    plt.xticks(np.arange(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(subjects)), subjects)
    plt.colorbar(label="NaN count")
    plt.tight_layout()
    plt.savefig(out_dir / "ALL_subjects_nan_columns_heatmap.png", dpi=220)
    plt.close()


# =========================
# MAIN
# =========================
def main() -> None:
    subject_files = list_subject_files(RAW_PROTOCOL_DIR)

    # exclude subjects
    for s in list(EXCLUDE_SUBJECTS):
        subject_files.pop(s, None)

    subjects = list(subject_files.keys())
    print(f"RAW_PROTOCOL_DIR: {RAW_PROTOCOL_DIR}")
    print(f"Subjects (post-exclude): {subjects}")
    print(f"Filter ACTIVITIES={ACTIVITIES} | EXCLUDE_ACTIVITIES={EXCLUDE_ACTIVITIES}\n")

    # per soggetto: activity counts & missingness
    per_subject_counts: Dict[str, pd.Series] = {}
    per_subject_nan: Dict[str, pd.Series] = {}

    for subject, fp in subject_files.items():
        df_raw = load_dat(fp)

        # filtri base (opzionali ma utili per capire il "dataset effettivo")
        df = apply_basic_filters(df_raw)

        # opzionale: se non vogliamo considerare HR nelle colonne problematiche
        if not CHECK_HR and "HR" in df.columns:
            df = df.drop(columns=["HR"])

        # counts attività
        counts = ensure_all_activities_indexed(activity_counts(df))
        per_subject_counts[subject] = counts

        # missingness
        n_nan, n_inf = compute_missingness(df)
        per_subject_nan[subject] = n_nan

        # stampa rapida
        missing_cls = counts[counts == 0].index.tolist() if ACTIVITIES is not None else []
        print(f"[{subject}] rows_raw={len(df_raw):>7} rows_filtered={len(df):>7} | missing_acts={missing_cls}")

        # grafici per soggetto
        plot_subject_activity_bar(subject, counts, OUT_DIR)
        plot_missingness_bar(subject, n_nan, n_inf, OUT_DIR)

    # --------
    # Heatmap globale soggetti x attività
    # --------
    if ACTIVITIES is not None:
        mat = np.stack([per_subject_counts[s].to_numpy(dtype=int) for s in subjects], axis=0)
        plot_activity_matrix_heatmap(subjects, mat, ACTIVITIES, OUT_DIR)

    # --------
    # Heatmap NaN: scegliamo le "top columns" più problematiche globalmente
    # --------
    # somma NaN per colonna su tutti i soggetti
    nan_sum = None
    for s in subjects:
        if nan_sum is None:
            nan_sum = per_subject_nan[s].copy()
        else:
            nan_sum = nan_sum.add(per_subject_nan[s], fill_value=0)

    nan_sum = nan_sum.sort_values(ascending=False)
    top_cols = nan_sum.head(25)  # top 25 colonne più NaN
    top_cols = top_cols[top_cols > 0]
    if len(top_cols) > 0:
        cols = [str(c) for c in top_cols.index.tolist()]
        mat_nan = np.stack(
            [per_subject_nan[s].reindex(top_cols.index).to_numpy(dtype=int) for s in subjects],
            axis=0,
        )
        plot_missing_cols_matrix(subjects, mat_nan, cols, OUT_DIR)
        print(f"\nTop NaN columns saved heatmap with {len(cols)} cols.")
    else:
        print("\nNo NaN columns detected globally (after filters).")

    print(f"\nSaved plots in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
