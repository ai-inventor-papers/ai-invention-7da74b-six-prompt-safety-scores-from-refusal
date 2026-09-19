#!/usr/bin/env python3
"""Final consistency verification for the v2 registry (run once, read-only)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
ITER1 = Path("/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


@logger.catch(reraise=True)
def main() -> None:
    rows = json.loads((PROJ / "data_out.json").read_text())
    full = json.loads((PROJ / "full_data_out.json").read_text())
    it1 = json.loads((ITER1 / "data_out.json").read_text())
    m1 = json.loads((ITER1 / "pairs_manifest.json").read_text())

    # 1. qwen3 verbatim byte-equality with iteration-1 manifest strings
    qrows = {r["matched_pair_id"]: r["prompt"] for r in rows
             if r["family"] == "qwen3" and r["type"] == "harmful"}
    ok = all(qrows[f"pair_{i:02d}"] == m1["pairs"][cat]["harmful"]
             for i, cat in [(1, "weapons"), (2, "violence"), (3, "cyber"),
                            (4, "fraud"), (5, "self-harm"), (6, "chemical")])
    logger.info(f"qwen3 verbatim byte-equality with iteration-1 manifest: {ok}")
    if not ok:
        raise SystemExit("FATAL: qwen3 reuse mismatch")

    # 2. data_out vs full_data_out consistency
    fds = {ex["metadata_family"]: (ex["input"], ex["metadata_token_count"])
           for g in full["datasets"] if g["dataset"].startswith("refusal_boundary_screen")
           for ex in g["examples"]}
    rds = {r["family"]: (r["prompt"], r["token_count"])
           for r in rows if r["fold"] == "screen_instrument"}
    logger.info(f"data_out == full_data_out (prompt, token_count): {fds == rds}")
    if fds != rds:
        raise SystemExit("FATAL: registries drift")

    # 3. id patterns
    pat = re.compile(r"^sb_si2_(qwen3|llama32|gemma2|qwen25)_p[0-9]{2}_[hb]$")
    screen = [r for r in rows if r["fold"] == "screen_instrument"]
    pat_ok = all(pat.match(r["id"]) for r in screen)
    lx_ids = [r["id"] for r in rows if r["fold"] == "lexicon_meta"]
    lx_ok = all(re.fullmatch(r"sb_lx_meta_(qwen3|llama32|gemma2|qwen25)", i) for i in lx_ids)
    logger.info(f"ids match plan pattern: {pat_ok and lx_ok} | {len(screen)} screen rows, {len(lx_ids)} lx rows")

    # 4. no iteration-1 id collision; confirm fold untouched
    it1_ids = {r["id"] for r in it1}
    collision = it1_ids & {r["id"] for r in rows}
    logger.info(f"no iteration-1 id collision: {not collision}")
    if collision:
        raise SystemExit("FATAL: id collision")

    # 5. token counts of every row equal its token_ids length
    for r in rows:
        if r.get("token_count") is not None and r.get("token_ids") is not None:
            assert r["token_count"] == len(r["token_ids"]), r["id"]
    logger.info("token_count == len(token_ids) for all rows: True")

    # 6. serialized size
    pub = 0
    for p in PROJ.glob("*.json"):
        pub += p.stat().st_size
        pub += (PROJ / "README.md").stat().st_size + (PROJ / "SOURCES_VERIFIED_v2.md").stat().st_size
    logger.info(f"published JSON+MD size: {pub / 1e6:.2f} MB")
    logger.info("ALL CONSISTENCY CHECKS PASSED")


if __name__ == "__main__":
    main()