#!/usr/bin/env python3
"""Recompute rows + selection + method_out.json from ALL evidence checkpoints.

CPU-only: loads each evidence JSON in results/evidence, recomputes metrics,
jackknife and the row dict, then runs the pre-hoc selection rule and writes
results/method_out.json + rows_summary.json.  Also supports a JSON
post-processing pass to move a top-level ``metrics_agg`` key into
``metadata.metrics_agg`` (schema compatibility with exp_gen_sol_out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import build_instrument  # noqa: E402
from method import (  # noqa: E402
    Metrics, _model_one_liner, selection_rule, _write_output, CANDIDATES,
)

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:%M:%S}|{level:<7}|{message}")


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev-dir", default=str(Path(__file__).resolve().parent / "results" / "evidence"))
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--postprocess-only", action="store_true",
                    help="only move top-level metrics_agg into metadata (fix prior write)")
    args = ap.parse_args()

    ev_dir = Path(args.ev_dir)
    out_dir = Path(args.out_dir)
    inst = build_instrument()

    if args.postprocess_only:
        out_path = out_dir / "method_out.json"
        data = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "metrics_agg" in data and "metadata" in data:
            data["metadata"]["metrics_agg"] = data.pop("metrics_agg")
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            logger.info(f"post-processed {out_path}: moved metrics_agg into metadata")
        return

    rows = []
    for ev_path in sorted(ev_dir.glob("*.json")):
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        if not ev.get("pairs"):
            logger.warning(f"skipping empty evidence {ev_path.name}")
            continue
        metrics = Metrics(ev, lexicon=inst.lexicon).compute_all()
        jk = {c: {} for c in CANDIDATES + ["B0", "B1", "B2"]}
        for ex in list(ev["pairs"].keys()):
            import method
            sub = method.jackknife_scores(ev, inst.lexicon, ex)
            for c in sub:
                jk[c][ex] = sub[c]
        spec = ev["spec"]
        row = {
            "model_key": f"{spec.get('family')}|{spec.get('role')}|{ev_path.stem.split('__')[-1]}",
            "family": spec.get("family", "?"),
            "role": spec.get("role", "?"),
            "repo_id": spec.get("repo_id", ev_path.stem),
            "spec": spec,
            "evidence": ev,
            "metrics": metrics,
            "jackknife": jk,
            "splice_max_abs_diff": float(ev.get("splice_max_abs_diff", -1.0)),
        }
        rows.append(row)
        logger.info(f"row {row['model_key']}: {_model_one_liner(row)}")

    logger.info(f"finalizing {len(rows)} rows")
    selection = selection_rule(rows)
    logger.info(f"Selection: survivors={selection.get('survivors')} winner={selection.get('winner')} "
                f"deception_ok={selection.get('deception_ok')}")

    # minimal args shim for _write_output (metadata is built inside it)
    class _A:
        out_dir = str(out_dir)
        cache_dir = "../cache"
        enable_c4 = True
    _write_output(_A(), inst, rows, selection, {}, {})


if __name__ == "__main__":
    main()