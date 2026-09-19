#!/usr/bin/env python3
"""Run 32 broad HF dataset searches in parallel; save compact results.

Todo 2: diverse BROAD queries across chosen source (HuggingFace catalog).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json, sys, time

import requests

from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

QUERIES = [
    # safety / harmful prompts
    "harmful prompts", "safety evaluation", "jailbreak prompts", "refusal responses",
    "llm safety benchmark", "red team prompts", "unsafe prompts", "toxic prompts",
    "malicious instructions", "attack prompts llm",
    # category-specific
    "self harm content", "cyber security threats", "fraud scam content",
    "weapons bomb instructions", "violence content moderation", "chemical synthesis dangerous",
    "illicit behavior", "prompt injection",
    # refusal / alignment
    "refusal dataset", "safety aligned responses", "ai refusal", "abliteration",
    "safety classifiers", "moderation dataset",
    # instruction corpora (benign)
    "alpaca instructions", "sharegpt conversations", "open assistant",
    "instruction following english", "dolly dataset", "user prompts english",
    # safety benchmarks
    "harmbench", "advbench", "safety aligned llm",
]

API = "https://huggingface.co/api/datasets"


def search(q: str) -> list[dict]:
    try:
        r = requests.get(API, params={"search": q, "limit": 10, "sort": "downloads"}, timeout=25)
        r.raise_for_status()
        out = []
        for d in r.json():
            out.append({
                "id": d.get("id"),
                "downloads": d.get("downloads"),
                "likes": d.get("likes"),
                "tags": d.get("tags", [])[:6],
                "desc": (d.get("description") or "")[:220],
            })
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning(f"search failed for {q!r}: {e}")
        return []


@logger.catch(reraise=True)
def main() -> None:
    t0 = time.time()
    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(search, q): q for q in QUERIES}
        for fut in as_completed(futs):
            q = futs[fut]
            results[q] = fut.result()
    out = Path("temp/searches.json")
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    logger.info(f"done in {time.time()-t0:.1f}s: {len(results)} queries -> {out}")


if __name__ == "__main__":
    main()