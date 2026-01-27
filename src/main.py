import argparse
import json

from src.pamap2_project.utils.config_loader import load_config
from src.pamap2_project.utils.run_manager import (
    copy_config_to_run,
    get_latest_run_dir,
    get_next_run_dir,
    get_run_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAMAP2 Project (style prof + runs)")

    parser.add_argument(
        "--config",
        type=str,
        default="src/pamap2_project/config/default.yaml",
        help="Path al file YAML di configurazione",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="print-config",
        choices=[
            "print-config",
            "windows",
            "split",
            "normalize",
            "train-rnn",
            "evaluate",
            "svm-features",
            "train-svm",
            "evaluate-svm",
        ],
        help="Stage della pipeline",
    )

    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Nome run (es: run_001). Se omesso, evaluate usa l'ultima run.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    # -----------------------------
    # PRINT CONFIG
    # -----------------------------
    if args.stage == "print-config":
        print("Configurazione caricata correttamente:\n")
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    # -----------------------------
    # WINDOWS
    # -----------------------------
    if args.stage == "windows":
        from src.pamap2_project.data.make_windows import run_make_windows

        run_make_windows(cfg)
        return

    # -----------------------------
    # SPLIT
    # -----------------------------
    if args.stage == "split":
        from src.pamap2_project.data.split_windows import run_split

        run_split(cfg)
        return

    # -----------------------------
    # NORMALIZE
    # -----------------------------
    if args.stage == "normalize":
        from src.pamap2_project.data.normalize import run_normalize

        run_normalize(cfg)
        return

    # -----------------------------
    # SVM FEATURES
    # -----------------------------
    if args.stage == "svm-features":
        from src.pamap2_project.data.window_features import run_make_svm_features

        run_make_svm_features(cfg)
        return

    # -----------------------------
    # TRAIN RNN
    # -----------------------------
    if args.stage == "train-rnn":
        run_dir = get_next_run_dir()
        copy_config_to_run(args.config, run_dir)
        print(f"\n[RUN] Created: {run_dir}\n")

        from src.pamap2_project.train.train_rnn import run_train_rnn

        run_train_rnn(cfg, run_dir)
        return

    # -----------------------------
    # EVALUATE RNN
    # -----------------------------
    if args.stage == "evaluate":
        if args.run is None:
            run_dir = get_latest_run_dir()
        else:
            run_dir = get_run_dir(args.run)

        print(f"\n[RUN] Using: {run_dir}\n")

        from src.pamap2_project.eval.evaluate_rnn import run_evaluate_rnn

        run_evaluate_rnn(cfg, run_dir)
        return

    # -----------------------------
    # TRAIN SVM
    # -----------------------------
    if args.stage == "train-svm":
        run_dir = get_next_run_dir()
        copy_config_to_run(args.config, run_dir)
        print(f"\n[RUN] Created: {run_dir}\n")

        from src.pamap2_project.train.train_svm import run_train_svm

        run_train_svm(cfg, run_dir)
        return

    # -----------------------------
    # EVALUATE SVM
    # -----------------------------
    if args.stage == "evaluate-svm":
        if args.run is None:
            run_dir = get_latest_run_dir()
        else:
            run_dir = get_run_dir(args.run)

        print(f"\n[RUN] Using: {run_dir}\n")

        from src.pamap2_project.eval.evaluate_svm import run_evaluate_svm

        run_evaluate_svm(cfg, run_dir)
        return


if __name__ == "__main__":
    main()
