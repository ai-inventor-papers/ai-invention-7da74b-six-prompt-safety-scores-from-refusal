#!/usr/bin/env python3
"""STEP 5b (fallback) - Greedy 8-token snippets via OpenRouter free Qwen3.

Local HF Qwen3-0.6B-Instruct was unavailable (HF resolver quota exhausted on
this egress IP for the whole artifact window), so snippets are generated with
the smallest available instruct model on OpenRouter's free tier
(qwen/qwen3.8-27b:free, temp=0, max_tokens=8, thinking disabled). Cost: $0.

Usage: OR_API_KEY=... python scripts/12_or_snippets.py
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

MODEL = "deepseek/deepseek-v4-flash-0731:free"
URL = "https://openrouter.ai/api/v1/chat/completions"
KEY = os.environ.get("OPENROUTER_API_KEY", "")


def gen_one(prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n"}],
        "max_tokens": 8,
        "temperature": 0,
        "reasoning": {"enabled": False},
    }
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    last = None
    for attempt in range(4):
        try:
            r = requests.post(URL, json=payload, headers=headers, timeout=90)
            if r.status_code == 200:
                j = r.json()
                content = (j.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                content = content.strip()
                if content:
                    return content
                last = "empty content: " + json.dumps(j)[:200]
            else:
                last = f"HTTP {r.status_code}: {r.text[:160]}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        logger.warning(f"attempt {attempt} failed ({last}); retry")
        time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"all attempts failed for {prompt[:40]!r}: {last}")


@logger.catch(reraise=True)
def main() -> None:
    assert KEY, "OPENROUTER_API_KEY not set"
    sel = json.loads(Path("temp/selected.json").read_text())
    prompts = [r["prompt"] for r in sel["lexicon_harmful"] + sel["lexicon_benign"]]
    logger.info(f"generating snippets for {len(prompts)} prompts via {MODEL}")
    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(gen_one, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
                logger.info(f"[{i:02d}] {prompts[i][:42]!r} -> {results[i][:42]!r}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"[{i:02d}] failed: {e}")
                results[i] = ""
    out = [{"index": i, "prompt": prompts[i], "snippet": results.get(i, ""),
            "prompt_tokens": 0, "new_tokens": 8} for i in range(len(prompts))]
    Path("temp/snippets.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    empty = [o["index"] for o in out if not o["snippet"]]
    logger.info(f"saved temp/snippets.json | empty snippets: {empty or 'none'}")


if __name__ == "__main__":
    main()