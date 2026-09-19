#!/usr/bin/env python3
"""STEP 8 - Final QC gates for data_out.json.

Checks: row counts (12/60/40/20); per-category confirm harmful == 10;
prompt shape (8-25 tokens, ASCII, no control chars); disjointness matrix
(instrument vs confirm vs lexicon: 0 exact, 0 normalized-substring, max
token-jaccard < 0.6); enums valid; schema validation; gold labels present
for lexicon fold. Writes temp/qc_report.json.
"""
from pathlib import Path
import json, re, sys

from loguru import logger
from transformers import AutoTokenizer
import jsonschema

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def tokenize(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


@logger.catch(reraise=True)
def main() -> None:
    rows = json.loads(Path("data_out.json").read_text())
    tok = AutoTokenizer.from_pretrained("temp/qwen3_tok")
    report: dict = {}
    problems: list[str] = []

    folds = {}
    for r in rows:
        folds.setdefault(r["fold"], []).append(r)
    report["counts"] = {k: len(v) for k, v in folds.items()}
    # exact composition
    si = [r for r in folds.get("screen_instrument", [])]
    cf_h = [r for r in folds.get("confirm", []) if r["type"] == "harmful"]
    cf_b = [r for r in folds.get("confirm", []) if r["type"] == "benign"]
    lx = [r for r in folds.get("lexicon_validation", [])]
    if not (len(si) == 12 and len(cf_h) == 60 and len(cf_b) == 40 and len(lx) == 20):
        problems.append(f"counts wrong: {len(si)}/{len(cf_h)}/{len(cf_b)}/{len(lx)}")
    logger.info(f"counts si={len(si)} cf_h={len(cf_h)} cf_b={len(cf_b)} lx={len(lx)}")

    from collections import Counter
    per_cat = Counter(r["category"] for r in cf_h)
    report["confirm_harmful_per_category"] = dict(sorted(per_cat.items()))
    if any(per_cat[c] != 10 for c in ["violence", "weapons", "cyber", "fraud", "self-harm", "chemical"]):
        problems.append(f"per-category != 10: {dict(per_cat)}")

    # prompt shape
    lengths = {"harmful": [], "benign": []}
    for r in rows:
        p = r["prompt"]
        if not p:
            problems.append(f"empty prompt {r['id']}")
        if any(ord(ch) < 32 for ch in p):
            problems.append(f"control char in {r['id']}")
        if not re.fullmatch(r"[ -~]+", p):
            problems.append(f"non-ASCII in {r['id']}: {p[:40]!r}")
        n = len(tokenize(tok, p))
        lengths[r["type"]].append(n)
        if r["type"] == "harmful" and not (8 <= n <= 25):
            problems.append(f"harmful token range {r['id']}: {n}")
        if r["type"] == "benign" and not (8 <= n <= 25):
            problems.append(f"benign token range {r['id']}: {n}")
    import statistics
    report["tokens"] = {
        "harmful": {"mean": round(statistics.mean(lengths["harmful"]), 2),
                    "median": statistics.median(lengths["harmful"]),
                    "min": min(lengths["harmful"]), "max": max(lengths["harmful"])},
        "benign": {"mean": round(statistics.mean(lengths["benign"]), 2),
                   "median": statistics.median(lengths["benign"]),
                   "min": min(lengths["benign"]), "max": max(lengths["benign"])},
    }

    # disjointness matrix across the 3 folds (prompt-level)
    def group(fold_rows):
        return [(r["prompt"], r["id"]) for r in fold_rows]

    g_si, g_cf, g_lx = group(si), group(cf_h + cf_b), group(lx)
    matrix = {}
    for name, g1, g2 in [("instrument-confirm", g_si, g_cf),
                         ("instrument-lexicon", g_si, g_lx),
                         ("confirm-lexicon", g_cf, g_lx)]:
        exact = sum(1 for p1, _ in g1 for p2, _ in g2 if p1 == p2)
        sub = 0
        worst_j = 0.0
        for p1, _ in g1:
            n1 = norm(p1)
            for p2, _ in g2:
                if p1 == p2:
                    continue
                n2 = norm(p2)
                if n1 and n2 and (n1 in n2 or n2 in n1):
                    sub += 1
                t1, t2 = set(tokenize(tok, p1)), set(tokenize(tok, p2))
                j = len(t1 & t2) / max(1, len(t1 | t2))
                worst_j = max(worst_j, j)
        matrix[name] = {"exact": exact, "normalized_substring": sub, "max_token_jaccard": round(worst_j, 3)}
        if exact or sub or worst_j >= 0.6:
            problems.append(f"disjointness FAIL {name}: {matrix[name]}")
        logger.info(f"disjoint {name}: {matrix[name]}")

    # within-confirm dedup (harmful and benign separately)
    for typ, gl in [("harmful", group(cf_h)), ("benign", group(cf_b))]:
        worst = 0.0
        for i in range(len(gl)):
            for j in range(i + 1, len(gl)):
                t1, t2 = set(tokenize(tok, gl[i][0])), set(tokenize(tok, gl[j][0]))
                jv = len(t1 & t2) / max(1, len(t1 | t2))
                worst = max(worst, jv)
        report[f"within_confirm_{typ}_max_jaccard"] = round(worst, 3)
        if worst >= 0.6:
            problems.append(f"within-confirm {typ} jaccard {worst:.3f}")

    # enums + gold labels
    for r in rows:
        if r["type"] not in ("harmful", "benign"):
            problems.append(f"bad type {r['id']}")
        if r["fold"] == "lexicon_validation" and r["gold_label"] not in ("refusal", "compliance"):
            problems.append(f"missing gold_label {r['id']}")
        if r["fold"] != "lexicon_validation" and r["gold_label"] is not None:
            problems.append(f"unexpected gold_label {r['id']}")
        if not re.fullmatch(r"^(pair_[0-9]{2}|)$", r["matched_pair_id"]):
            problems.append(f"bad matched_pair_id {r['id']}")
    if any(r["fold"] == "screen_instrument" and r["matched_pair_id"] == "" for r in si):
        problems.append("instrument row missing pair id")

    # schema validation
    schema = json.loads(Path("schema.json").read_text())
    try:
        jsonschema.validate(rows, schema)
        report["schema_validation"] = "PASS"
    except jsonschema.ValidationError as e:
        report["schema_validation"] = f"FAIL: {e.message}"
        problems.append(str(e))

    report["problems"] = problems
    report["all_pass"] = not problems
    Path("temp/qc_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"QC problems: {len(problems)} | all_pass={report['all_pass']}")
    for p in problems[:10]:
        logger.error(p)


if __name__ == "__main__":
    main()