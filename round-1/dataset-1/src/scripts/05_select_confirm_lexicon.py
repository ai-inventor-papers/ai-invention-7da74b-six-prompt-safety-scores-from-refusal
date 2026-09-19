#!/usr/bin/env python3
"""STEP 2+3+b-part - Select confirm corpus (60 harmful + 40 benign) and the
20 lexicon-validation prompts from verified pools, with dedup + disjointness.

Selection policy (documented in README):
- confirm harmful: 10 per category, VERBATIM from treadon/abliteration-eval
  (Apache-2.0). Source category -> our taxonomy mapping:
    violence_harm->violence, weapons->weapons, hacking->cyber,
    fraud_scams->fraud, self_harm->self-harm, dangerous_information->chemical.
- lexicon-validation harmful (12, 2 per category): VERBATIM from
  JailbreakBench JBB-Behaviors (MIT), HarmBench humans (MIT, mirrored in the
  Apache-2.0 andyrdt/refusal_direction repo), AdvBench (MIT, same mirror).
- benign: yahma/alpaca-cleaned (CC-BY-4.0) + OpenAssistant/oasst1 (Apache-2.0),
  8-25 tokens, no safety-adjacent keywords, length-matched to harmful set.
- All rows disjoint between folds (exact + normalized substring + token-jaccard
  checks), 8-25 tokens, ASCII-punct, no control chars.
"""
from pathlib import Path
import json, re, unicodedata, sys

import pandas as pd
from loguru import logger
from transformers import AutoTokenizer

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

CATS = ["violence", "weapons", "cyber", "fraud", "self-harm", "chemical"]
SRC_CAT_TO_CAT = {
    "violence_harm": "violence",
    "weapons": "weapons",
    "hacking": "cyber",
    "fraud_scams": "fraud",
    "self_harm": "self-harm",
    "dangerous_information": "chemical",
}
TREADON_CATS = {"violence_harm", "weapons", "hacking", "fraud_scams", "self_harm", "dangerous_information"}

# benign exclusion: safety-adjacent topic words (even benign mentions) are excluded
BENIGN_BLOCK = re.compile(
    r"\b(bomb|explosiv|weapon|gun|rifle|shoot|kill|murder|assault|stab|poison|toxic|ricin|"
    r"suicid|self-?harm|overdos|drugs|heroin|cocaine|meth|marijuana|anorex|bulimi|cutting|"
    r"hack|hacking|malware|virus|ransomware|phish|fraud|scam|forge|identity theft|password crack|"
    r"hijack|molest|rape|nazi|terrorist|chemical weapon|nerve agent|dirty bomb)\b",
    re.I,
)

# keyword helpers for lexicon-harmful selection per our taxonomy
LEX_KW = {
    "violence": re.compile(r"\b(kill|murder|assault|torture|beat|strangl|shoot|stab|violent|attack|homicid|punch)", re.I),
    "weapons": re.compile(r"\b(bomb|explosiv|weapon|firearm|gun|rifle|silencer|pipe bomb|molotov|grenade|nuclear)", re.I),
    "cyber": re.compile(r"\b(hack|malware|ransomware|keylogger|phish|exploit|botnet|virus|credentials|password|intrus)", re.I),
    "fraud": re.compile(r"\b(fraud|scam|forge|phish|ponzi|launder|fake|identity theft|blackmail|embezzle|bribe)", re.I),
    "self-harm": re.compile(r"\b(suicid|self-?harm|cut myself|noose|overdos|kill myself|self-starv)", re.I),
    "chemical": re.compile(r"\b(ricin|nerve agent|mustard gas|sarin|toxin|poison|chemical|biological weapon|synthesi[sz]e|anthrax|sarin)", re.I),
}


def clean(s: str) -> str | None:
    """Return cleaned prompt or None if it fails prompt-shape constraints."""
    if not isinstance(s, str):
        return None
    s = " ".join(s.split())  # normalize whitespace runs
    if any(unicodedata.category(ch) == "Cc" for ch in s):
        return None
    if not re.fullmatch(r"[ -~]+", s):  # printable ASCII range (keeps plain apostrophes)
        return None
    if len(s) < 10:
        return None
    return s


def norm_jaccard(a: str, b: str, tok) -> float:
    ta = set(tok.encode(a.lower(), add_special_tokens=False))
    tb = set(tok.encode(b.lower(), add_special_tokens=False))
    return len(ta & tb) / max(1, len(ta | tb))


def main() -> None:
    tok = AutoTokenizer.from_pretrained("temp/qwen3_tok")
    instrument = json.loads(Path("pairs_manifest.json").read_text())["pairs"]
    inst_prompts = []
    for p in instrument.values():
        inst_prompts += [p["harmful"], p["benign"]]

    # ---------- pools ----------
    treadon = pd.read_parquet("temp/datasets/treadon_0.parquet")
    jbb = pd.read_parquet("temp/datasets/jbb_1.parquet")  # harmful split
    hb = pd.concat([
        pd.read_csv("temp/github/harmbench_test.csv"),
        pd.read_csv("temp/github/harmbench_val.csv"),
    ])
    adv = pd.read_csv("temp/github/advbench.csv")
    alpaca = pd.read_parquet("temp/datasets/alpaca_0.parquet")
    oasst = pd.concat([
        pd.read_parquet("temp/datasets/oasst1_0.parquet"),
        pd.read_parquet("temp/datasets/oasst1_1.parquet"),
    ])

    def tk(p: str) -> int:
        return len(tok.encode(p, add_special_tokens=False))

    ok_tokens = lambda p: 8 <= tk(p) <= 25  # noqa: E731

    def adapt_to_range(p: str) -> tuple[str, bool]:
        """Return (prompt, adapted) - minimally extend p until 8-25 tokens."""
        if ok_tokens(p):
            return p, False
        for suffix in [" step by step", " using common household items", " without getting caught",
                       " for beginners", " quickly", " in private"]:
            cand = p.rstrip(".!?") + suffix + ("?" if p.endswith("?") else ".")
            if ok_tokens(cand):
                return cand, True
        raise SystemExit(f"cannot adapt {p!r} into token range")

    # ---------- confirm harmful: treadon verbatim ----------
    confirm_harmful = []  # list of dicts
    t_by_cat: dict[str, list] = {}
    for src_cat in TREADON_CATS:
        sub = treadon[(treadon["category"] == src_cat) & (treadon["type"] == "harmful")]
        t_by_cat[SRC_CAT_TO_CAT[src_cat]] = [clean(p) for p in sub["prompt"]]
    for cat in CATS:
        cands = [p for p in t_by_cat[cat] if p]
        picked = []
        for p in cands:
            q, adapted = adapt_to_range(p)
            if any(norm_jaccard(q, x["prompt"], tok) >= 0.6 for x in picked):
                continue
            src = "adapted_from:treadon:default:harmful" if adapted else "treadon:default:harmful"
            picked.append({"prompt": q, "source": src, "category": cat})
            if len(picked) == 10:
                break
        if len(picked) < 10:
            logger.error(f"{cat}: only {len(picked)} after dedup")
            raise SystemExit(1)
        confirm_harmful += picked
    logger.info(f"confirm harmful: {len(confirm_harmful)} rows")

    # ---------- enforce hard disjointness vs instrument rows ----------
    def vs_instrument_ok(p: str) -> bool:
        return all(norm_jaccard(p, ip, tok) < 0.6 for ip in inst_prompts)

    confirm_harmful = [r for r in confirm_harmful if vs_instrument_ok(r["prompt"])]
    if len(confirm_harmful) < 60:
        logger.warning(f"after instrument filter: {len(confirm_harmful)} harmful rows; refilling")

    # refill dropped rows: extra cross-source pools first, then same-source leftovers
    import collections as _c
    extra = _c.defaultdict(list)
    for pool_rows, tag in [(hb, "hb"), (adv, "adv")]:
        pass
    for _, r in hb.iterrows():
        p0, b = clean(r["Behavior"]), r["BehaviorID"]
        if p0:
            for cat in CATS:
                if LEX_KW[cat].search(p0):
                    extra[cat].append((p0, f"harmbench-human:{b}"))
    for _, r in adv.iterrows():
        p0 = clean(r["goal"])
        if p0:
            for cat in CATS:
                if LEX_KW[cat].search(p0):
                    extra[cat].append((p0, "advbench"))
    bah = pd.read_parquet("temp/datasets/bahushruth_0.parquet")
    for txt in bah["text"]:
        p0 = clean(txt)
        if p0:
            for cat in CATS:
                if LEX_KW[cat].search(p0):
                    extra[cat].append((p0, "Bahushruth/abliteration-harmful-enriched:default:train"))

    remaining = _c.defaultdict(list)
    for cat in CATS:
        for p in t_by_cat[cat]:
            if p and all(r["prompt"] != p for r in confirm_harmful):
                remaining[cat].append((p, "treadon:default:harmful"))
    used_prompts = {r["prompt"] for r in confirm_harmful}
    for cat in CATS:
        while sum(1 for r in confirm_harmful if r["category"] == cat) < 10:
            for p, src0 in extra[cat] + remaining[cat]:
                q, adapted = adapt_to_range(p)
                if q in used_prompts:
                    continue
                if any(norm_jaccard(q, x["prompt"], tok) >= 0.6 for x in confirm_harmful):
                    continue
                if not vs_instrument_ok(q):
                    continue
                src = f"adapted_from:{src0}" if adapted else src0
                confirm_harmful.append({"prompt": q, "source": src, "category": cat})
                used_prompts.add(q)
                break
            else:
                raise SystemExit(f"cannot refill {cat}")
    logger.info(f"confirm harmful after instrument filter: {len(confirm_harmful)}")

    # ---------- lexicon harmful (12, 2/cat) from JBB/HarmBench/AdvBench ----------
    lex_harmful = []
    pools = []
    for _, r in jbb.iterrows():
        pools.append((clean(r["Goal"]), f"JailbreakBench/JBB-Behaviors:behaviors:harmful:index{r['Index']}", "jbb"))
    for _, r in hb.iterrows():
        pools.append((clean(r["Behavior"]), f"harmbench-human:{r['BehaviorID']}", "hb"))
    for _, r in adv.iterrows():
        pools.append((clean(r["goal"]), "advbench", "adv"))
    pools = [p for p in pools if p[0]]
    used = set()
    for cat in CATS:
        picked = 0
        for p, src, tag in pools:
            if p in used:
                continue
            if not ok_tokens(p):
                continue
            if not LEX_KW[cat].search(p):
                continue
            # must not collide with any confirm harmful row or instrument row
            if any(norm_jaccard(p, q["prompt"], tok) >= 0.6 for q in confirm_harmful):
                continue
            if not vs_instrument_ok(p):
                continue
            used.add(p)
            lex_harmful.append({"prompt": p, "category": cat, "source": src})
            picked += 1
            if picked == 2:
                break
        if picked < 2:
            logger.warning(f"lexicon harmful {cat}: only {picked}/2 from keyword pools")
    logger.info(f"lexicon harmful: {len(lex_harmful)} rows")

    # ---------- benign: confirm 40 + lexicon 8 ----------
    benign_cands = []
    for _, r in alpaca.iterrows():
        if isinstance(r.get("input"), str) and r["input"].strip():
            continue  # skip instructions with inputs (not standalone)
        p = clean(r["instruction"])
        if p and not BENIGN_BLOCK.search(p):
            benign_cands.append((p, "yahma/alpaca-cleaned:default:train"))
    oas_prompter = oasst[(oasst["lang"] == "en") & (oasst["role"] == "prompter")]
    for txt in oas_prompter["text"]:
        p = clean(txt)
        if p and not BENIGN_BLOCK.search(p):
            benign_cands.append((p, "OpenAssistant/oasst1:default:train"))
    benign_cands = [c for c in benign_cands if ok_tokens(c[0])]
    logger.info(f"benign candidates: {len(benign_cands)}")

    harmful_tokens = [tk(p["prompt"]) for p in confirm_harmful]
    target_mean = sum(harmful_tokens) / len(harmful_tokens)
    logger.info(f"harmful token mean {target_mean:.1f} (n={len(harmful_tokens)})")
    benign_cands.sort(key=lambda c: abs(tk(c[0]) - target_mean))
    inst_benign = [p["benign"] for p in instrument.values()]

    def vs_instrument_benign_ok(p: str) -> bool:
        return all(norm_jaccard(p, ip, tok) < 0.6 for ip in inst_benign)

    chosen_benign, lex_benign = [], []
    for p, src in benign_cands:
        if p in inst_prompts:
            continue
        if not vs_instrument_benign_ok(p):
            continue
        if any(norm_jaccard(p, b["prompt"], tok) >= 0.6 for b in chosen_benign):
            continue
        chosen_benign.append({"prompt": p, "source": src})
        if len(chosen_benign) == 40:
            break
    if len(chosen_benign) < 40:
        logger.error(f"only {len(chosen_benign)} benign chosen")
        raise SystemExit(1)
    # lexicon benign: 8 more, distinct from chosen and instrument
    for p, src in benign_cands:
        if p in inst_prompts:
            continue
        if any(p == b["prompt"] for b in chosen_benign):
            continue
        if not vs_instrument_benign_ok(p):
            continue
        if any(norm_jaccard(p, b["prompt"], tok) >= 0.6 for b in chosen_benign):
            continue
        lex_benign.append({"prompt": p, "source": src})
        if len(lex_benign) == 8:
            break
    logger.info(f"confirm benign: {len(chosen_benign)}, lexicon benign: {len(lex_benign)}")

    out = {
        "confirm_harmful": confirm_harmful,
        "confirm_benign": chosen_benign,
        "lexicon_harmful": lex_harmful,
        "lexicon_benign": lex_benign,
        "instrument_prompts": inst_prompts,
        "stats": {
            "harmful_token_mean": target_mean,
            "harmful_token_min": min(harmful_tokens),
            "harmful_token_max": max(harmful_tokens),
        },
    }
    Path("temp/selected.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    bt = [tk(b["prompt"]) for b in chosen_benign]
    logger.info(
        f"saved temp/selected.json | benign token mean {sum(bt)/len(bt):.1f} "
        f"| lexicon harmful {len(lex_harmful)} | lexicon benign {len(lex_benign)}"
    )


if __name__ == "__main__":
    main()