#!/usr/bin/env python3
"""STEP 5/7e - QC gates for data_out.json v2 -> temp/qc_report_v2.json.

Gates (all must pass except documented fallbacks):
  5a  per-pair token equality under the family's own tokenizer (hard gate) and
      8-25 token range per member (no special tokens; iteration-1 convention).
  5b  ASCII-only (byte regex over 0x00-0x7F), no control chars, byte-exactness
      after JSON round-trip (load -> json.dump(ensure_ascii=False) -> identical).
  5c  within-registry dedup (no duplicate rows); cross-family reuse is logged
      separately (not a failure).
  5d  disjointness vs iteration-1 reserved CONFIRM fold (60+40): 0 exact, 0
      normalized-substring, max token-jaccard < 0.6; same vs the iteration-1
      lexicon_validation fold. Vs the iteration-1 SCREEN instrument, byte
      identity is EXPECTED for qwen3 (verbatim reuse) and canonical-reuse rows
      (deliberate continuity) and is reported, not failed.
  5e  alignment: shared prefix AND suffix >= 2 token IDs each; pairs whose
      alignment_gate == 'count_equality_only' (terminal-punctuation shape under
      the plan template bank) are recorded as documented fallbacks (plan 9.3).
  5f  lexicon decode sanity: decode each claimed Arditi Table 4 start token
      under the family tokenizer; 'I'-likeness recorded, discrepancies flagged.
Schema validation of data_out.json against schema.json v2 included.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
ITER1 = Path("/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")

from loguru import logger  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import jsonschema  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(PROJ / "logs" / "05_qc.log"), rotation="10 MB", level="DEBUG")

FAMILIES = ["qwen3", "llama32", "gemma2", "qwen25"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def jaccard(tok, a: str, b: str) -> float:
    ta, tb = set(tok.encode(a, add_special_tokens=False)), set(tok.encode(b, add_special_tokens=False))
    return len(ta & tb) / max(1, len(ta | tb))


@logger.catch(reraise=True)
def main() -> None:
    rows = json.loads((PROJ / "data_out.json").read_text())
    it1_rows = json.loads((ITER1 / "data_out.json").read_text())
    it1_confirm = [r for r in it1_rows if r["fold"] == "confirm"]
    it1_lx = [r for r in it1_rows if r["fold"] == "lexicon_validation"]
    it1_screen = [r for r in it1_rows if r["fold"] == "screen_instrument"]
    toks = {fam: AutoTokenizer.from_pretrained(str(PROJ / "temp" / "tokenizers" / fam),
                                               trust_remote_code=True, use_fast=True) for fam in FAMILIES}
    problems: list[str] = []
    report: dict = {}

    screen = [r for r in rows if r["fold"] == "screen_instrument"]
    lxmeta = [r for r in rows if r["fold"] == "lexicon_meta"]
    report["counts"] = {"screen_instrument_v2": len(screen), "lexicon_meta": len(lxmeta),
                        "total": len(rows)}
    if len(screen) != 48 or len(lxmeta) != 4:
        problems.append(f"row counts wrong: {len(screen)} screen / {len(lxmeta)} lx_meta")

    # ---- 5a/5b/5c per family ----
    per_family: dict[str, dict] = {}
    all_prompts: list[str] = [r["prompt"] for r in screen if r["prompt"]]
    for fam in FAMILIES:
        tok = toks[fam]
        fam_rows = [r for r in screen if r["family"] == fam]
        counts = {"harmful": [], "benign": []}
        pair_eq = {"pairs_checked": 0, "pairs_equal": 0}
        bad_ascii, bad_ctrl, bad_roundtrip = [], [], []
        # pair up by matched_pair_id
        by_pair: dict[str, dict[str, str]] = {}
        for r in fam_rows:
            by_pair.setdefault(r["matched_pair_id"], {})[r["type"]] = r["prompt"]
        for pnum, members in sorted(by_pair.items()):
            h, b = members["harmful"], members["benign"]
            th, tb = tok.encode(h, add_special_tokens=False), tok.encode(b, add_special_tokens=False)
            pair_eq["pairs_checked"] += 1
            if len(th) == len(tb):
                pair_eq["pairs_equal"] += 1
            else:
                problems.append(f"5a FAIL {fam} {pnum}: {len(th)} vs {len(tb)}")
            counts["harmful"].append(len(th))
            counts["benign"].append(len(tb))
            if not (8 <= len(th) <= 25 and 8 <= len(tb) <= 25):
                problems.append(f"5a range FAIL {fam} {pnum}: {len(th)}/{len(tb)}")
        for r in fam_rows:
            p = r["prompt"]
            if not p or any(ord(ch) < 32 for ch in p):
                bad_ctrl.append(r["id"])
            if not re.fullmatch(r"[ -~]+", p):
                bad_ascii.append(r["id"])
            if json.loads(json.dumps(p, ensure_ascii=False)) != p:
                bad_roundtrip.append(r["id"])
        per_family[fam] = {
            "pairs_checked": pair_eq["pairs_checked"],
            "pairs_equal": pair_eq["pairs_equal"],
            "token_range_harmful": [min(counts["harmful"]), max(counts["harmful"])],
            "token_range_benign": [min(counts["benign"]), max(counts["benign"])],
            "ascii_violations": bad_ascii, "control_char_violations": bad_ctrl,
            "roundtrip_violations": bad_roundtrip,
        }
        if bad_ascii or bad_ctrl or bad_roundtrip:
            problems.append(f"5b FAIL {fam}: ascii={bad_ascii} ctrl={bad_ctrl} rt={bad_roundtrip}")
        logger.info(f"{fam}: pairs {pair_eq} range H{per_family[fam]['token_range_harmful']} "
                    f"B{per_family[fam]['token_range_benign']} ascii_bad={len(bad_ascii)}")

    # 5c within-registry dedup: no duplicate prompt rows (same family+type+pair)
    seen: set[tuple] = set()
    dup = []
    for r in screen:
        key = (r["family"], r["type"], r["matched_pair_id"], r["prompt"])
        if key in seen:
            dup.append(r["id"])
        seen.add(key)
    report["dedup"] = {"duplicate_rows": dup}
    if dup:
        problems.append(f"5c FAIL duplicate rows: {dup}")

    # ---- 5d disjointness vs iteration-1 confirm fold (per family tokenizer) ----
    disc: dict[str, dict] = {}
    it1_confirm_strings = [r["prompt"] for r in it1_confirm]
    it1_lx_strings = [r["prompt"] for r in it1_lx]
    it1_screen_strings = [r["prompt"] for r in it1_screen]
    for fam in FAMILIES:
        tok = toks[fam]
        fam_prompts = [r["prompt"] for r in screen if r["family"] == fam]
        res = {"vs_confirm_fold": {}, "vs_lexicon_validation": {}, "vs_iter1_screen_instrument": {}}
        for label, pool in (("vs_confirm_fold", it1_confirm_strings),
                            ("vs_lexicon_validation", it1_lx_strings),
                            ("vs_iter1_screen_instrument", it1_screen_strings)):
            exact = sub = 0
            worst = 0.0
            for p in fam_prompts:
                if p in pool:
                    exact += 1
                n = norm(p)
                if n and any(n in norm(q) or norm(q) in n for q in pool):
                    sub += 1
                worst = max(worst, max((jaccard(tok, p, q) for q in pool), default=0.0))
            res[label] = {"exact": exact, "normalized_substring": sub,
                          "max_token_jaccard": round(worst, 3)}
            if label == "vs_iter1_screen_instrument":
                # byte identity with the iteration-1 screen instrument is EXPECTED
                # for qwen3 (verbatim) and canonical-reuse rows: continuity, not
                # contamination; only NON-screen collisions would be a failure.
                if exact != len(set(fam_prompts) & set(it1_screen_strings)):
                    problems.append(f"5d unexpected screen-exact count {fam}: {exact}")
                res[label]["expected_reuse"] = exact
            else:
                if exact or sub or worst >= 0.6:
                    problems.append(f"5d FAIL {fam} {label}: {res[label]}")
        disc[fam] = res
        logger.info(f"{fam} disjoint: cf={res['vs_confirm_fold']} lx={res['vs_lexicon_validation']} "
                    f"screen_reuse={res['vs_iter1_screen_instrument']['exact']}")
    report["disjointness"] = disc

    # ---- 5e alignment ----
    align_res = []
    for fam in FAMILIES:
        fam_rows = [r for r in screen if r["family"] == fam]
        by_pair = {}
        for r in fam_rows:
            by_pair.setdefault(r["matched_pair_id"], {})[r["type"]] = r
        for pnum in sorted(by_pair):
            h = by_pair[pnum]["harmful"]
            gate = h.get("notes") or ""
            th = h["token_ids"]
            tb = by_pair[pnum]["benign"]["token_ids"]
            assert len(th) == len(tb)
            n = len(th)
            pref = next((i for i in range(n) if th[i] != tb[i]), n)
            last = next((i for i in range(n - 1, -1, -1) if th[i] != tb[i]), -1)
            suf = n - 1 - last
            full = pref >= 2 and suf >= 2
            align_res.append({"family": fam, "pair": pnum,
                              "prefix_shared": pref, "suffix_shared": suf,
                              "alignment_gate": "full" if full else "count_equality_only"})
            documented = ("count_equality_only" in gate) or ("STEP 9.3" in gate)
            if not full and not documented:
                problems.append(f"5e FAIL {fam} {pnum}: prefix={pref} suffix={suf} not documented")
    report["alignment"] = align_res
    n_full = sum(1 for a in align_res if a["alignment_gate"] == "full")
    logger.info(f"alignment: {n_full}/24 full, {len(align_res) - n_full} documented count-equality-only")

    # ---- 5f lexicon decode sanity ----
    lx_decode: dict[str, dict] = {}
    for r in lxmeta:
        tok = toks[r["family"]]
        dec = [tok.decode([t]) for t in r["token_ids"]]
        i_like = all(("I" in d or d.strip().startswith("As")) for d in dec)
        lx_decode[r["family"]] = {"token_ids": r["token_ids"], "decoded": dec,
                                  "i_like": i_like}
        if not i_like:
            problems.append(f"5f NOTE {r['family']}: decoded {dec} not 'I'-like (documented, kept)")
    report["lexicon_decode"] = lx_decode

    # ---- schema validation (v2) ----
    schema = json.loads((PROJ / "schema.json").read_text())
    try:
        jsonschema.validate(rows, schema)
        report["schema_validation"] = "PASS"
    except jsonschema.ValidationError as e:
        report["schema_validation"] = f"FAIL: {e.message}"
        problems.append(f"schema FAIL: {e.message}")

    report["problems"] = problems
    report["all_pass"] = not problems
    (PROJ / "temp" / "qc_report_v2.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"QC v2: {len(problems)} problems | all_pass={report['all_pass']}")
    for p in problems[:12]:
        logger.error(p)


if __name__ == "__main__":
    main()