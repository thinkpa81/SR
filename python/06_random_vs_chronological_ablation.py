
from __future__ import annotations

import argparse
import shutil
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_lib.modeling import ensure_dir, fit_base_and_hybrid, load_analysis_base, setup_logging, split_chronological, split_random_exact
from pipeline_lib.paths import resolve_project_dir

DEFAULT_PROJECT_DIR = resolve_project_dir(Path(__file__).resolve().parent)

                                                                                  
                                                                            
DEFAULT_RANDOM_SEED = 42
RANDOM_VALID_N = 25961
RANDOM_TEST_N = 27063


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivot = df[df["dataset"] == "test"].pivot(index="model", columns="split_scheme", values=metric).sort_index()
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(9, 5))
    ax.set_title(f"{metric} comparison: chronological vs random split")
    ax.set_ylabel(metric)
    ax.set_xlabel("Model")
    ax.legend(title="Split scheme")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random-split vs chronological-split ablation for KLIPS SR manuscript.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--n-valid", type=int, default=RANDOM_VALID_N)
    parser.add_argument("--n-test", type=int, default=RANDOM_TEST_N)
    parser.add_argument(
        "--run-random-refit",
        action="store_true",
        help="Run the full random-split model refit. The default reuses cached chronological predictions for routine submission validation.",
    )
    args = parser.parse_args()

    project_dir = args.project_dir
    output_dir = project_dir / "sr_additional_analyses" / "ablation_random_vs_chronological"
    ensure_dir(output_dir)
    log_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(output_dir / "logs" / f"08_random_vs_chronological_ablation_{log_stamp}.log")

    cached_metrics = project_dir / "outputs_klips_sr" / "stage3_hybrid_metrics.csv"
    cached_valid = project_dir / "outputs_klips_sr" / "stage3_valid_predictions_with_hybrid.csv"
    cached_test = project_dir / "outputs_klips_sr" / "stage3_test_predictions_with_hybrid.csv"

    if not args.run_random_refit:
                                                                                     
        missing = [path for path in [cached_metrics, cached_valid, cached_test] if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Cached chronological predictions are missing. Run 03_train_hybrid_bootstrap.py first, "
                f"or call this script with --run-random-refit. Missing: {missing}"
            )
        chrono_metrics = pd.read_csv(cached_metrics).rename(columns={"split": "dataset"})
        chrono_metrics["split_scheme"] = "chronological"
        chrono_valid_pred = pd.read_csv(cached_valid)
        chrono_test_pred = pd.read_csv(cached_test)
        all_metrics = chrono_metrics
        all_metrics.to_csv(output_dir / "ablation_split_comparison_metrics.csv", index=False, encoding="utf-8-sig")
        chrono_valid_pred.to_csv(output_dir / "chronological_valid_predictions.csv", index=False, encoding="utf-8-sig")
        chrono_test_pred.to_csv(output_dir / "chronological_test_predictions.csv", index=False, encoding="utf-8-sig")
        summary = all_metrics[all_metrics["dataset"] == "test"].copy()
        summary.to_csv(output_dir / "ablation_test_summary_for_manuscript.csv", index=False, encoding="utf-8-sig")
        manifest = {
            "input_file": str(project_dir / "outputs_klips_sr" / "processed" / "analysis_base_with_label.csv"),
            "mode": "cached_chronological_only",
            "outputs": [
                "ablation_split_comparison_metrics.csv",
                "ablation_test_summary_for_manuscript.csv",
                "chronological_valid_predictions.csv",
                "chronological_test_predictions.csv",
            ],
            "note": "Routine submission validation reuses cached chronological predictions from the primary pipeline. Run with --run-random-refit to regenerate the computationally expensive random-split ablation.",
        }
        with open(output_dir / "ablation_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return

                                                        
    df = load_analysis_base(project_dir)
    chrono_train, chrono_valid, chrono_test = split_chronological(df)
    random_train, random_valid, random_test = split_random_exact(df, n_valid=args.n_valid, n_test=args.n_test, seed=args.seed)

    chrono_metrics, chrono_valid_pred, chrono_test_pred = fit_base_and_hybrid(chrono_train, chrono_valid, chrono_test, "chronological")
    random_metrics, random_valid_pred, random_test_pred = fit_base_and_hybrid(random_train, random_valid, random_test, "random")

    all_metrics = pd.concat([chrono_metrics, random_metrics], ignore_index=True)
    all_metrics.to_csv(output_dir / "ablation_split_comparison_metrics.csv", index=False, encoding="utf-8-sig")
    chrono_valid_pred.to_csv(output_dir / "chronological_valid_predictions.csv", index=False, encoding="utf-8-sig")
    chrono_test_pred.to_csv(output_dir / "chronological_test_predictions.csv", index=False, encoding="utf-8-sig")
    random_valid_pred.to_csv(output_dir / "random_valid_predictions.csv", index=False, encoding="utf-8-sig")
    random_test_pred.to_csv(output_dir / "random_test_predictions.csv", index=False, encoding="utf-8-sig")

    summary = all_metrics[all_metrics["dataset"] == "test"].pivot(index="model", columns="split_scheme", values=["roc_auc", "pr_auc", "brier", "recall_at_20", "lift_at_20"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    for metric in ["roc_auc", "pr_auc", "recall_at_20", "lift_at_20"]:
        if f"{metric}_chronological" in summary.columns and f"{metric}_random" in summary.columns:
            summary[f"{metric}_random_minus_chronological"] = summary[f"{metric}_random"] - summary[f"{metric}_chronological"]
    if "brier_chronological" in summary.columns and "brier_random" in summary.columns:
        summary["brier_random_minus_chronological"] = summary["brier_random"] - summary["brier_chronological"]
    summary.to_csv(output_dir / "ablation_test_summary_for_manuscript.csv", index=False, encoding="utf-8-sig")

    plot_metric(all_metrics, "roc_auc", output_dir / "figure_ablation_roc_auc.png")
    plot_metric(all_metrics, "pr_auc", output_dir / "figure_ablation_pr_auc.png")
    plot_metric(all_metrics, "brier", output_dir / "figure_ablation_brier.png")
    plot_metric(all_metrics, "lift_at_20", output_dir / "figure_ablation_lift20.png")
                                                                       
    shutil.copyfile(output_dir / "figure_ablation_roc_auc.png", output_dir / "FigS3_random_split_roc_auc.png")
    shutil.copyfile(output_dir / "figure_ablation_lift20.png", output_dir / "FigS4_random_split_lift20.png")

    manifest = {
        "input_file": str(project_dir / "outputs_klips_sr" / "processed" / "analysis_base_with_label.csv"),
        "mode": "random_refit",
        "seed": args.seed,
        "random_valid_n": args.n_valid,
        "random_test_n": args.n_test,
        "outputs": [
            "ablation_split_comparison_metrics.csv", "ablation_test_summary_for_manuscript.csv",
            "chronological_valid_predictions.csv", "chronological_test_predictions.csv",
            "random_valid_predictions.csv", "random_test_predictions.csv",
            "figure_ablation_roc_auc.png", "figure_ablation_pr_auc.png",
            "figure_ablation_brier.png", "figure_ablation_lift20.png",
            "FigS3_random_split_roc_auc.png", "FigS4_random_split_lift20.png",
        ],
        "note": "Random split reuses person-wave rows across partitions by design and is intended only as an optimism-ablation benchmark against the chronological split.",
    }
    with open(output_dir / "ablation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
