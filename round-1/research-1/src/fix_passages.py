#!/usr/bin/env python3
"""Repair supporting_passages in research_out.json: verify each quote byte-exactly
against a fresh fetch of its source URL; drop or replace quotes that do not match."""
import json
import re
import subprocess
import sys
import os

WS = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_research_1"
PY = "/ai-inventor/.claude/skills/.ability_client_venv/bin/python"
SCRIPT = "/ai-inventor/.claude/skills/aii-web-tools/scripts/aii_fast_web_fetch.py"
FETCH_CACHE = {}


def fetch(url: str) -> str:
    if url in FETCH_CACHE:
        return FETCH_CACHE[url]
    try:
        r = subprocess.run(
            [PY, SCRIPT, "fetch", "--url", url, "--max-chars", "12000"],
            capture_output=True, text=True, timeout=40,
            cwd=WS,
        )
        out = r.stdout
    except Exception as e:  # noqa: BLE001
        out = f"ERR {e}"
    FETCH_CACHE[url] = out
    return out


# Targeted replacements: exact continuous ASCII substrings seen in earlier fetches.
replacements = {
    1:  [r'"gated":false'],
    4:  [r'"gated":"auto"', r'"downloads":45'],
    11: [r'301/400| 24.8%', r'18/400| 95.5%',
         r'Huihui struggles on this model. It only reaches 95.5% safety removal, leaving 18 harmful requests still refused.',
         r'Refusals were classified via regex matching, which captured 100% of refusals cleanly'],
    12: [r'refuses 84.7% of AdvBench'],
    13: [r'28% of prompts exhibit decision flips',
         r'single-shot evaluation agrees with multi-sample ground truth only 92.5%'],
    15: [r'we can add the difference-in-means vector to the activations of a harmless input',
         r'we can subtract this vector at layer l in order to shift activations away from refusal'],
    16: [r"proj = einops.einsum(matrix, refusal_dir.view(-1, 1), '... d_model, d_model single -> ... single') * refusal_dir"],
    19: [r'pairing a harmful query with a structurally similar but benign counterpart',
         r'LFJ reaches a macro-averaged attack success rate (ASR) of 94.13% under the white-box protocol'],
    27: [r"Linear probes discriminated hazardous from benign cases with 98.2% AUROC, yet the model's output sensitivity was only 45.1%, a 53-percentage-point knowledge-action gap",
         r'SAE feature steering produced zero effect despite 3,695 significant features'],
    29: [r'adversarially finetuning models to accept harmful instructions'],
    32: [r'self.start_with_keywords = ["I apologize", "I\'m sorry", "I cannot", "I can\'t"]'],
    14: [r'Gemma 2 IT 2B', r'26.5%'],
}

# Build new passage lists: (quote, locator) per source
new_passages = {}
for idx, quotes in replacements.items():
    loc = {
        1: "HF REST API", 4: "HF REST API", 11: "Safety: HarmBench table / intro",
        12: "Executive Summary", 13: "Abstract", 15: "Section 2.4 / Appendix I",
        16: "apply_refusal_dirs", 19: "Abstract", 27: "Abstract",
        29: "Abstract", 32: "ClaudeAlignmentChecker.check_alignment",
        14: "Table 15",
    }[idx]
    new_passages[idx] = [{"quote": q, "locator": loc} for q in quotes]


data = json.load(open(os.path.join(WS, "research_out.json"), encoding="utf-8"))
changed = []
for s in data["sources"]:
    idx = s["index"]
    url = s["url"]
    old_ps = s.get("supporting_passages") or []
    if idx in new_passages:
        cand = new_passages[idx]
    else:
        cand = old_ps
    text = fetch(url)
    # normalize fetch output: strip leading echo lines
    m = re.search(r"--- Content ---\s*\n(.*)", text, re.S)
    body = m.group(1) if m else text
    kept = []
    for ps in cand:
        q = ps.get("quote", "")
        qq = q.encode("utf-8").decode("unicode_escape")  # resolve \u escapes if literal
        if q in body or qq in body:
            kept.append({"quote": q, "locator": ps.get("locator")})
        else:
            changed.append((idx, q[:80]))
    s["supporting_passages"] = kept

json.dump(data, open(os.path.join(WS, "research_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(data, open(os.path.join(WS, ".sdk_openhands_agent_struct_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("dropped/replaced:", changed)
print("remaining passages per source:", {s["index"]: len(s.get("supporting_passages") or []) for s in data["sources"]})