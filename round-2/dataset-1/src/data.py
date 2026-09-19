#!/usr/bin/env python3
"""data.py - Standardize the tokenizer-agnostic registry to exp_sel_data_out.

Run:  uv run data.py

Pipeline position (v2 dataset artifact):
  scripts/02_build_pairs.py   -> temp/pairs_v2_draft.json   (per-family pair evidence,
                                                             token counts/ids, alignment,
                                                             search trails)
  scripts/03_lexicon_meta.py  -> temp/lexicon_meta_v2.json  (per-family Arditi Table 4
                                                             start-token sets, live decode)
  scripts/04_assemble.py      -> data_out.json (52-row canonical registry),
                                 schema.json (v2, additive), full_data_out.json
                                 (exp_sel_data_out view, 5 dataset groups),
                                 pairs_manifest_v2.json
  THIS data.py                -> idempotent re-run of the assembly step via
                                 scripts/04_assemble.py so that `uv run data.py`
                                 reproducibly produces the standardized outputs from
                                 the temp evidence (no tokenization needed here).

The 4 chosen DATASETS (groups in full_data_out.json):
  refusal_boundary_screen_instrument_{qwen3,llama32,gemma2,qwen25} (12 examples each;
  one example per prompt row; input=prompt, output=byte-exact matched pair counterpart)
  + refusal_boundary_lexicon_meta (4 examples; input='').
Every row of the canonical registry is exactly one example (no fold-level folding);
all per-example information is carried in flat metadata_* fields.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


def main() -> None:
    assemble = Path(__file__).resolve().parent / "scripts" / "04_assemble.py"
    logger.info(f"running assembly: {assemble}")
    runpy.run_path(str(assemble), run_name="__main__")
    logger.success("data.py done: data_out.json + full_data_out.json + schema.json + pairs_manifest_v2.json regenerated")


if __name__ == "__main__":
    main()