#!/usr/bin/env python3
"""Race the HF resolver window: download Qwen3-0.6B-Instruct files right after
each 5-min quota reset. Files: config, tokenizer set, model.safetensors(+inx).
Reuses whatever is already on disk (temp/qwen3_06b_instruct)."""
import json, subprocess, sys, time
from pathlib import Path

import requests

BASE = "https://huggingface.co/Qwen/Qwen3-0.6B-Instruct/resolve/main"
DEST = Path("temp/qwen3_06b_instruct")
DEST.mkdir(parents=True, exist_ok=True)
need = ["config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "model.safetensors", "model.safetensors.index.json"]
ok = set()
for f in need:
    if (DEST / f).exists() and (DEST / f).stat().st_size > 1000:
        ok.add(f)


def fetch(f: str) -> bool:
    if f in ok:
        return True
    dest = DEST / f
    try:
        r = requests.get(f"{BASE}/{f}", timeout=180 if f.startswith("model") else 45)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            ok.add(f)
            print(f"OK {f} {len(r.content)}", flush=True)
            return True
        print(f"FAIL {f} {r.status_code}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ERR {f} {e}", flush=True)
    return False


def main() -> None:
    from concurrent.futures import ThreadPoolExecutor
    for attempt in range(14):
        # if index exists, fetch listed shards too
        if "model.safetensors.index.json" in ok:
            try:
                idx = json.loads((DEST / "model.safetensors.index.json").read_text())
                for sh in sorted(set(idx["weight_map"].values())):
                    fetch(sh)
                if all((DEST / sh).stat().st_size > 1000 for sh in set(idx["weight_map"].values())):
                    print("ALL_SHARDS_OK", flush=True)
                    return
            except Exception as e:  # noqa: BLE001
                print(f"idx err {e}", flush=True)
        # parallel burst: all files at once
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, need))
        done = len(ok)
        print(f"attempt {attempt}: {done}/{len(need)} files ok", flush=True)
        if done == len(need):
            print("ALL_FILES_OK", flush=True)
            return
        try:
            h = requests.head(f"{BASE}/config.json", timeout=15)
            rl = h.headers.get("ratelimit", "t=300")
            t = int(rl.split("t=")[1].split(";")[0]) + 3
            print(f"sleeping {t}s till window reset", flush=True)
            time.sleep(max(t, 10))
        except Exception:  # noqa: BLE001
            time.sleep(120)


if __name__ == "__main__":
    main()