#!/usr/bin/env python3
"""STEP 1 - Source verification gate.

For every candidate repo: exists? gated? license (exact)? configs/splits/num_rows?
feature names? downloads/likes. Record into temp/gate_results.json and print a table.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, sys

import requests
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

CANDIDATES = [
    # plan priority chain (harmful)
    "cais/harmbench_behaviors",
    "bench-llm/OR-Bench",
    "jailbreakbench/JBB-Behaviors",
    "longyan99/Safety-Prompts",
    "longyan97/Safety-Prompts",
    # searches extras (harmful / jailbreak)
    "rubend18/ChatGPT-Jailbreak-Prompts",
    "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "Bahushruth/abliteration-harmful-enriched",
    "treadon/abliteration-eval",
    "hirundo-io/refusal-responses",
    "deepset/prompt-injections",
    # benign instruction corpora
    "yahma/alpaca-cleaned",
    "OpenAssistant/oasst1",
    "databricks/databricks-dolly-15k",
    "tatsu-lab/alpaca",
    "anon8231489123/ShareGPT_Vicuna_unfiltered",
]

API = "https://huggingface.co/api/datasets"
INFO = "https://datasets-server.huggingface.co/info"


def gate(rid: str) -> dict:
    rec = {"id": rid}
    try:
        r = requests.get(f"{API}/{rid}", timeout=25)
        rec["api_status"] = r.status_code
        if r.status_code == 200:
            j = r.json()
            rec["gated"] = j.get("gated")
            rec["license"] = (j.get("cardData") or {}).get("license")
            rec["downloads"] = j.get("downloads")
            rec["likes"] = j.get("likes")
            rec["tags"] = (j.get("tags") or [])[:10]
        else:
            return rec
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"api: {e}"
        return rec
    try:
        j = requests.get(INFO, params={"dataset": rid}, timeout=25).json()
        if "dataset_info" not in j:
            rec["info_error"] = j.get("error")
            return rec
        rec["configs"] = {}
        for cfg, cj in (j["dataset_info"] or {}).items():
            splits = (cj or {}).get("splits") or {}
            rec["configs"][cfg] = {
                "splits": [f"{s['name']}:{s.get('num_examples')}" for s in splits.values()],
                "features": sorted((cj.get("features") or {}).keys()),
            }
    except Exception as e:  # noqa: BLE001
        rec["info_error"] = f"info: {e}"
    return rec


@logger.catch(reraise=True)
def main() -> None:
    results = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(gate, r): r for r in CANDIDATES}
        for fut in as_completed(futs):
            rid = futs[fut]
            results[rid] = fut.result()
    Path("temp/gate_results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    for rid, rec in sorted(results.items()):
        cfg = rec.get("configs") or {}
        cfgs = ",".join(f"{k}({','.join(v['splits'][:2])})" for k, v in list(cfg.items())[:3])
        logger.info(
            f"{rid:45s} api={rec.get('api_status')} gated={rec.get('gated')} "
            f"lic={rec.get('license')} dl={rec.get('downloads')} likes={rec.get('likes')} "
            f"cfgs={cfgs or rec.get('info_error')}"
        )


if __name__ == "__main__":
    main()