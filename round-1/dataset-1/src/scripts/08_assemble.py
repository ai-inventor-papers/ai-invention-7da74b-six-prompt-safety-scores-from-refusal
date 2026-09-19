#!/usr/bin/env python3
"""STEP 6 - Assemble data_out.json (132 rows) from pairs_manifest,
temp/selected.json, temp/snippets.json; write schema.json; validate."""
from pathlib import Path
import json, sys

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

CATS = ["violence", "weapons", "cyber", "fraud", "self-harm", "chemical"]
# plan order for instrument pair numbering: pair_01 weapons .. pair_06 chemical
PAIR_CATS = ["weapons", "violence", "cyber", "fraud", "self-harm", "chemical"]
VALID_CATS = set(CATS) | {"benign"}


def build_rows(manifest, selected, snippets):
    rows = []
    # --- fold=screen_instrument: 12 rows from 6 pairs ---
    pair_no = 0
    for cat in PAIR_CATS:
        pair_no += 1
        p = manifest["pairs"][cat]
        pair_id = f"pair_{pair_no:02d}"
        for hb, typ in [("harmful", "h"), ("benign", "b")]:
            rows.append({
                "id": f"sb_si_p{pair_no:02d}_{hb[0]}",
                "prompt": p[hb],
                "type": hb,
                "category": cat if hb == "harmful" else "benign",
                "fold": "screen_instrument",
                "source": "template:screen_instrument",
                "matched_pair_id": pair_id,
                "output": "",
                "gold_label": None,
            })
    # --- fold=confirm: 60 harmful + 40 benign ---
    cat_abbrev = {"violence": "viol", "weapons": "weap", "cyber": "cybe",
                  "fraud": "fraud", "self-harm": "self", "chemical": "chem"}
    counts = {}
    for r in selected["confirm_harmful"]:
        cat = r["category"]
        n = counts.get(cat, 0) + 1
        counts[cat] = n
        rows.append({
            "id": f"sb_cf_{cat_abbrev[cat]}_{n:02d}",
            "prompt": r["prompt"], "type": "harmful", "category": cat,
            "fold": "confirm", "source": r["source"], "matched_pair_id": "",
            "output": "", "gold_label": None,
        })
    for i, r in enumerate(selected["confirm_benign"], start=1):
        rows.append({
            "id": f"sb_cf_ben_{i:02d}", "prompt": r["prompt"], "type": "benign",
            "category": "benign", "fold": "confirm", "source": r["source"],
            "matched_pair_id": "", "output": "", "gold_label": None,
        })
    # --- fold=lexicon_validation: 20 rows with snippets + gold labels ---
    snip_by_prompt = {s["prompt"]: s for s in snippets}
    lex_rows = selected["lexicon_harmful"] + selected["lexicon_benign"]
    for i, r in enumerate(lex_rows, start=1):
        sn = snip_by_prompt[r["prompt"]]
        rows.append({
            "id": f"sb_lx_val_{i:02d}", "prompt": r["prompt"],
            "type": r.get("category", "benign") != "benign" and "harmful" or "benign",
            "category": r.get("category", "benign"),
            "fold": "lexicon_validation",
            "source": r["source"] + " | snippet: generated:openrouter:deepseek/deepseek-v4-flash-0731:free (temp=0, max_tokens=8)",
            "matched_pair_id": "", "output": sn["snippet"], "gold_label": "PENDING",
        })
    return rows


def main() -> None:
    manifest = json.loads(Path("pairs_manifest.json").read_text())
    selected = json.loads(Path("temp/selected.json").read_text())
    snippets = json.loads(Path("temp/snippets.json").read_text())
    rows = build_rows(manifest, selected, snippets)
    # deterministic order: fold, then id
    fold_order = {"screen_instrument": 0, "confirm": 1, "lexicon_validation": 2}
    rows.sort(key=lambda r: (fold_order[r["fold"]], r["id"]))
    Path("data_out.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    from collections import Counter
    logger.info(f"total rows: {len(rows)} | folds: {dict(Counter(r['fold'] for r in rows))}")
    logger.info(f"confirm harmful per cat: {dict(Counter(r['category'] for r in rows if r['fold']=='confirm' and r['type']=='harmful'))}")
    logger.info("saved data_out.json (gold_label=PENDING until labeled)")


if __name__ == "__main__":
    main()