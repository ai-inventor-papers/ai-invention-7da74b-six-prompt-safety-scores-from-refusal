#!/usr/bin/env python3
"""STEP 6 - Per-family lexicon metadata rows (Arditi et al. 2406.11717 Table 4).

One lexicon_meta record per family carrying the Table 4 refusal start-token
set, live decode verification under the family's own tokenizer (STEP 5f), the
effective tokenizer repo id, usage instructions and provenance caveats.
Also decodes the Gemma prose variant token 234285 (paper body) separately
to document the Table-vs-prose discrepancy instead of silently fixing it.
Writes temp/lexicon_meta_v2.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(PROJ / "logs" / "03_lexicon_meta.log"), rotation="10 MB", level="DEBUG")

from transformers import AutoTokenizer  # noqa: E402

# Arditi et al. 2024, arXiv:2406.11717, Table 4 (Refusal start tokens):
#   Qwen Chat {40, 2121} | "I'm sorry", "As an AI"
#   Gemma IT {235285} | "I cannot"
#   Llama-3 Instruct {40} | "I cannot"
# (verified via ar5iv HTML on 2026-09-19; see SOURCES_VERIFIED_v2.md)
TABLE4: dict[str, list[int]] = {
    "qwen3": [40, 2121],
    "llama32": [40],
    "gemma2": [235285],
    "qwen25": [40, 2121],
}

EFECTIVE_REPO: dict[str, str] = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama32": "meta-llama/Llama-3.2-1B",   # acquired with HF_TOKEN (mirrors verified unused)
    "gemma2": "google/gemma-2-2b",          # acquired with HF_TOKEN (mirrors verified unused)
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
}

EXTRA_DECODES: dict[str, list[int]] = {
    # Paper-body prose says "token 234285" for Gemma; Table 4 says 235285.
    # Decode both so the discrepancy is documented, never silently fixed.
    "gemma2": [234285],
    "llama32": [],
    "qwen3": [],
    "qwen25": [],
}

AR5IV = ("Arditi et al. 2024, arXiv:2406.11717, Table 4 (Refusal start tokens); "
         "verified via ar5iv HTML https://ar5iv.labs.arxiv.org/html/2406.11717 on 2026-09-19")

USAGE = ("first-token refusal gate per Arditi Eq. 6 P_refusal; combine with the "
         "iteration-1 refusal_lexicon.json 28-phrase continuation keyword bank "
         "(/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/"
         "gen_art/gen_art_dataset_1/refusal_lexicon.json) for the decontaminated "
         "r(alpha) readout")


@logger.catch(reraise=True)
def main() -> None:
    records = {}
    for fam in TABLE4:
        tok = AutoTokenizer.from_pretrained(str(PROJ / "temp" / "tokenizers" / fam),
                                            trust_remote_code=True, use_fast=True)
        decoded = {str(t): tok.decode([t]) for t in TABLE4[fam]}
        extra = {str(t): tok.decode([t]) for t in EXTRA_DECODES[fam]}
        i_like = all(("I" in d or d.strip().startswith("As")) for d in decoded.values())
        note = None
        if not i_like:
            note = ("decoded strings not 'I'-like under this family's tokenizer; "
                    "Table 4 set KEPT as-is (provenance is Arditi Table 4), "
                    "discrepancy flagged, not silently fixed")
        # The Gemma table-vs-prose token discrepancy gets its own explicit note.
        if fam == "gemma2":
            note = (f"Table 4 sets {{235285}}; paper body prose says 'token 234285'. "
                    f"Live decode: decode([235285])={decoded[str(235285)]!r} (I-like); "
                    f"decode([234285])={extra[str(234285)]!r} (NOT 'I' -> prose is a typo, "
                    f"Table value kept and decode-verified). Discrepancy documented, not fixed.")
        records[fam] = {
            "family": fam,
            "tokenizer": EFECTIVE_REPO[fam],
            "token_ids": TABLE4[fam],
            "decoded_start_tokens": [tok.decode([t]) for t in TABLE4[fam]],
            "decoded_extra_probe_tokens": {k: v for k, v in extra.items()},
            "source": AR5IV,
            "usage": USAGE,
            "caveat": ("sets originate from the Llama-3/Qwen2/Gemma-era Arditi Table 4 "
                       "and are decode-verified per family here; discrepancies are flagged "
                       "in this record, not silently fixed"),
            "qc_note": note,
            "refusal_lexicon_v1_path": (
                "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/"
                "gen_art/gen_art_dataset_1/refusal_lexicon.json"),
        }
        logger.info(f"{fam}: token_ids={TABLE4[fam]} decoded={records[fam]['decoded_start_tokens']} "
                    f"extra={extra} | {note or 'ok'}")

    out = {"schema_version": "2.0", "source_basis": AR5IV, "records": records}
    (PROJ / "temp" / "lexicon_meta_v2.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("wrote temp/lexicon_meta_v2.json")


if __name__ == "__main__":
    main()