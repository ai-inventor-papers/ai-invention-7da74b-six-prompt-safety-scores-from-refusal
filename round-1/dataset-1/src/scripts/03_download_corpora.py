#!/usr/bin/env python3
"""Download parquet shards for the 4 chosen corpora into temp/datasets/."""
import json, subprocess, sys
from pathlib import Path
import requests

JOBS = [
    ("bench-llm/or-bench", "or-bench-80k", "orbench"),
    ("JailbreakBench/JBB-Behaviors", "behaviors", "jbb"),
    ("yahma/alpaca-cleaned", "default", "alpaca"),
    ("OpenAssistant/oasst1", "default", "oasst1"),
]

outdir = Path("temp/datasets")
outdir.mkdir(parents=True, exist_ok=True)

for rid, cfg, short in JOBS:
    u = f"https://huggingface.co/api/datasets/{rid}/parquet/{cfg}"
    j = requests.get(u, timeout=30).json()
    # response format: {"train": ["url0", "url1", ...], "test": [...]}
    urls = []
    for split_urls in j.values():
        if isinstance(split_urls, list):
            urls.extend(split_urls)
    print(f"{short}: {len(urls)} shards")
    for i, url in enumerate(urls):
        dest = outdir / f"{short}_{i}.parquet"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  skip existing {dest.name}")
            continue
        subprocess.run(["curl", "-sL", "--max-time", "300", "-o", str(dest), url], check=True)
        print(f"  {dest.name} {dest.stat().st_size//1024} KB")