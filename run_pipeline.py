"""
run_pipeline.py — End-to-end pipeline for siads699 capstone.

Steps:
  1. Generate PLM embeddings (train + holdout) — skips files that already exist
  2. Build model-ready datasets (merge embeddings with cleaned data)
  3. Cross-validated ensemble modeling (CV metrics + weighted + stacked ensemble)
  4. Final holdout evaluation (GDPa3, all PLMs × all models)

Run from anywhere:
    uv run run_pipeline.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))       # enables: from src.models import ...
sys.path.insert(0, str(ROOT / "src"))  # enables: from CDR_work import ...


def _banner(step_n, label):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  STEP {step_n}: {label}")
    print(f"{bar}\n")


def _run(step_n, label, fn, hard_fail=False):
    """
    Run a pipeline step, catching and reporting errors cleanly.

    hard_fail=True  — abort the whole pipeline on failure (steps 1–2, since
                      later steps depend on their outputs).
    hard_fail=False — warn and continue (steps 3–5 are independent enough
                      that a failure in one shouldn't block the others).
    """
    _banner(step_n, label)
    t0 = time.time()
    try:
        fn()
        print(f"\n  [STEP {step_n} OK — {(time.time() - t0) / 60:.1f} min]")
    except Exception as exc:
        print(f"\n  [STEP {step_n} FAILED] {type(exc).__name__}: {exc}")
        if hard_fail:
            raise RuntimeError(
                f"Step {step_n} ({label}) failed — cannot continue."
            ) from exc
        print("  Continuing with next step...\n")


def main():
    t_start = time.time()

    _run(1, "Generate PLM embeddings (train + holdout)",
         lambda: __import__("src.pipelines.generate_embeddings", fromlist=["main"]).main(),
         hard_fail=True)

    _run(2, "Build model-ready datasets",
         lambda: __import__("src.pipelines.build_model_datasets", fromlist=["main"]).main(),
         hard_fail=True)

    _run(3, "Cross-validated ensemble modeling",
         lambda: __import__("src.prediction_modeling.ensemble_beta", fromlist=["driver"]).driver())

    _run(4, "Final holdout evaluation",
         lambda: __import__("src.prediction_modeling.final_holdout_eval", fromlist=["main"]).main())

    _run(5, "SHAP feature importance analysis",
         lambda: __import__("src.prediction_modeling.shap_analysis", fromlist=["main"]).main())

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete — {elapsed:.1f} min total")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
