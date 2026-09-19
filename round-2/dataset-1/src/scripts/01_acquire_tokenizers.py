#!/usr/bin/env python3
"""STEP 2 - Acquire the four family tokenizers (files only, never weights).

For each family, try primary repo -> fallback chain (per artifact plan STEP 2),
download via AutoTokenizer.from_pretrained(trust_remote_code=True, use_fast=True),
and save a local copy to temp/tokenizers/{family}/ so re-runs are offline.
Retries: tenacity exponential backoff for HF egress/resolver quota errors.
Writes: temp/tokenizers/acquired.json + SOURCES_VERIFIED_v2.md gating table.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJ / "temp" / "hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJ / "temp" / "hf_cache" / "hub"))

from loguru import logger  # noqa: E402
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(PROJ / "logs" / "01_acquire.log"), rotation="10 MB", level="DEBUG")

from transformers import AutoTokenizer  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

# Family -> ordered candidate repo ids (primary first). Gating re-verified live
# 2026-09-19 via the HF REST API (see SOURCES_VERIFIED_v2.md); gated primaries
# are skipped automatically unless HF_TOKEN is present in the environment.
FAMILIES: dict[str, list[str]] = {
    "qwen3": ["Qwen/Qwen3-0.6B"],
    "llama32": [
        "meta-llama/Llama-3.2-1B",          # gated=manual; needs HF_TOKEN
        "unsloth/Llama-3.2-1B-Instruct",    # ungated fallback (plan primary fallback)
        "NousResearch/Llama-3.2-1B",        # ungated; byte-size-identical tokenizer to gated original
        "unsloth/Llama-3.2-1B",             # ungated fallback (plan alternate)
    ],
    "gemma2": [
        "google/gemma-2-2b",                # gated=manual; needs HF_TOKEN
        "google/gemma-2-2b-it",             # gated=manual; needs HF_TOKEN
        "unsloth/gemma-2-2b-it",            # ungated fallback; tokenizer byte-size-identical
    ],
    "qwen25": [
        "Qwen/Qwen2.5-1.5B-Instruct",       # gated=False (verified live)
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Base",
    ],
}

# Repos downloaded for cross-checks (not a family tokenizer): kept in their own
# cache dir so they are never confused with the effective family tokenizers.
CROSS_CHECKS: dict[str, list[str]] = {
    "llama32_faithful_mirror": ["NousResearch/Llama-3.2-1B"],
}


def probe_gating(api: HfApi, repo_id: str) -> dict:
    """Live gating probe via the HF REST API (read-only)."""
    import requests
    url = f"https://huggingface.co/api/models/{repo_id}"
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "ai-inventor-run"})
        if r.status_code == 200:
            d = r.json()
            return {"repo": repo_id, "gated": str(d.get("gated")), "downloads": d.get("downloads"),
                    "private": d.get("private"), "tags": (d.get("tags") or [])[:6]}
        return {"repo": repo_id, "gated": f"HTTP{r.status_code}", "downloads": None, "private": None,
                "tags": [], "note": "API error"}
    except Exception as e:  # noqa: BLE001
        return {"repo": repo_id, "gated": f"ERR:{type(e).__name__}", "downloads": None,
                "private": None, "tags": [], "note": str(e)[:120]}


def _resolver_quota(e: Exception) -> bool:
    """HF egress/resolver quota errors are retryable."""
    text = str(e)
    return any(k in text.lower() for k in ("402", "quota", "rate limit", "rate_limit", "503", "429", "etimedout", "connection", "resolve"))


@retry(retry=retry_if_exception_type(Exception), wait=wait_exponential(multiplier=3, min=6, max=120),
       stop=stop_after_attempt(6), reraise=False)
def _download(repo_id: str, out_dir: Path) -> Path:
    try:
        tok = AutoTokenizer.from_pretrained(
            repo_id, trust_remote_code=True, use_fast=True, local_files_only=False
        )
    except Exception as e:  # noqa: BLE001
        if _resolver_quota(e):
            logger.warning(f"retryable HF error for {repo_id}: {type(e).__name__} {str(e)[:120]}")
            raise
        raise
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir, safe_serialization=True)
    return out_dir


@logger.catch(reraise=True)
def main() -> None:
    tok_dir = PROJ / "temp" / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    gating = []
    for fam, repos in FAMILIES.items():
        for rid in repos:
            gating.append(probe_gating(api, rid))
            logger.info(f"gating {rid}: {gating[-1]}")
    for rid in sorted({r for rs in CROSS_CHECKS.values() for r in rs}):
        gating.append(probe_gating(api, rid))
        logger.info(f"gating {rid}: {gating[-1]}")
    gating = sorted(gating, key=lambda d: (d["repo"], str(d.get("note", ""))))

    acquired: dict[str, dict] = {}
    for fam, repos in FAMILIES.items():
        local = tok_dir / fam
        if (local / "tokenizer_config.json").exists() and (local / "tokenizer.json").exists():
            # offline re-run: reuse local copy
            acquired[fam] = {"effective_repo": f"local:{local.name}", "local_dir": str(local),
                             "blocked_reason": None, "attempted": repos}
            logger.info(f"{fam}: using local copy {local}")
            continue
        attempted: list[str] = []
        blocked: str | None = None
        for rid in repos:
            attempted.append(rid)
            g = next((x for x in gating if x["repo"] == rid), {})
            if str(g.get("gated")) in ("manual", "True", "auto") and not os.environ.get("HF_TOKEN"):
                logger.warning(f"{fam}: {rid} gated={g.get('gated')} and no HF_TOKEN -> skip")
                blocked = f"{rid} gated={g.get('gated')}, no HF_TOKEN (fallback chain continued)"
                continue
            try:
                out = _download(rid, local)
                acquired[fam] = {"effective_repo": rid, "local_dir": str(out),
                                 "blocked_reason": None, "attempted": attempted}
                logger.success(f"{fam}: acquired tokenizer from {rid} -> {out}")
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"{fam}: {rid} FAILED: {type(e).__name__} {str(e)[:200]}")
                blocked = f"{rid} failed: {type(e).__name__}: {str(e)[:160]}"
                shutil.rmtree(local, ignore_errors=True)
                continue
        if fam not in acquired:
            acquired[fam] = {"effective_repo": None, "local_dir": None,
                             "blocked_reason": blocked or "all candidates failed", "attempted": attempted}
            logger.error(f"{fam}: ALL CANDIDATES FAILED -> {blocked}")

    # cross-check downloads (faithful mirror comparison for llama32)
    for name, repos in CROSS_CHECKS.items():
        local = tok_dir / name
        if (local / "tokenizer.json").exists():
            logger.info(f"cross-check {name}: local copy present")
            continue
        for rid in repos:
            g = next((x for x in gating if x["repo"] == rid), {})
            if str(g.get("gated")) in ("manual", "True", "auto") and not os.environ.get("HF_TOKEN"):
                logger.warning(f"cross-check {name}: {rid} gated -> skip")
                continue
            try:
                _download(rid, local)
                logger.success(f"cross-check {name}: {rid} -> {local}")
                break
            except Exception as e:  # noqa: BLE001
                logger.error(f"cross-check {name}: {rid} FAILED {type(e).__name__}: {str(e)[:160]}")

    out = {"families": acquired, "gating": gating}
    (tok_dir / "acquired.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("wrote temp/tokenizers/acquired.json")
    for fam, a in acquired.items():
        logger.info(f"{fam}: effective_repo={a['effective_repo']}")


if __name__ == "__main__":
    main()