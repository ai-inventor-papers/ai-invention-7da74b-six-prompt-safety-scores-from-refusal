#!/usr/bin/env python3
"""STEP 7 - Assemble data_out.json (52 rows), schema.json v2, full_data_out.json
(exp_sel_data_out view, 5 groups) and pairs_manifest_v2.json.

Inputs: temp/pairs_v2_draft.json (pair search evidence), temp/lexicon_meta_v2.json,
temp/tokenizers/acquired.json, and the iteration-1 pairs_manifest.json (for the
verbatim-qwen3 provenance framing).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]

from loguru import logger  # noqa: E402

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(PROJ / "logs" / "04_assemble.log"), rotation="10 MB", level="DEBUG")

ITER1 = Path("/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_dataset_1")
CAT2PAIR = {"weapons": "pair_01", "violence": "pair_02", "cyber": "pair_03",
            "fraud": "pair_04", "self-harm": "pair_05", "chemical": "pair_06"}
ORDER = ["weapons", "violence", "cyber", "fraud", "self-harm", "chemical"]
FAMILIES = ["qwen3", "llama32", "gemma2", "qwen25"]
EFFECTIVE_REPO = {
    "qwen3": "Qwen/Qwen3-0.6B",
    "llama32": "meta-llama/Llama-3.2-1B",
    "gemma2": "google/gemma-2-2b",
    "qwen25": "Qwen/Qwen2.5-1.5B-Instruct",
}
AR5IV = ("Arditi et al. 2024, arXiv:2406.11717, Table 4 (Refusal start tokens); "
         "verified via ar5iv HTML https://ar5iv.labs.arxiv.org/html/2406.11717 "
         "on 2026-09-19")


def build_schema_v2() -> dict:
    """Additive extension of the iteration-1 schema (old rows stay valid)."""
    s = json.loads((ITER1 / "schema.json").read_text())
    s["$id"] = "https://ai-inventor.local/schemas/refusal_boundary_data_out.schema.v2.json"
    s["title"] = "Refusal-boundary test prompts and lexicon registry (v2, tokenizer-agnostic)"
    s["description"] = (
        "One row per prompt. Folds: screen_instrument (6 matched harmful/benign pairs PER "
        "tokenizer family x4: qwen3/llama32/gemma2/qwen25), confirm (reserved 60 harmful + "
        "40 benign), lexicon_validation (20 prompts with greedy 8-token output snippets and "
        "gold refusal/compliance labels), lexicon_meta (per-family Arditi Table 4 refusal "
        "start-token sets). v2 extends the v1 type/category/fold enums and adds optional "
        "per-row tokenization metadata; v1 rows remain valid."
    )
    items = s["items"]
    props = items["properties"]
    # extend enums
    props["type"]["enum"] = ["harmful", "benign", "lexicon_meta"]
    props["category"]["enum"] = ["violence", "weapons", "cyber", "fraud", "self-harm",
                                 "chemical", "benign", "lexicon"]
    props["fold"]["enum"] = ["screen_instrument", "confirm", "lexicon_validation", "lexicon_meta"]
    # extend id pattern (additive: backreferences-style alternation for the v2 ids)
    props["id"]["pattern"] = (
        "^(sb_si_p[0-9]{2}_[hb]|sb_cf_(viol|weap|cybe|fraud|self|chem)_[0-9]{2}|"
        "sb_cf_ben_[0-9]{2}|sb_lx_val_[0-9]{2}|"
        "sb_si2_(qwen3|llama32|gemma2|qwen25)_p[0-9]{2}_[hb]|"
        "sb_lx_meta_(qwen3|llama32|gemma2|qwen25))$"
    )
    # prompt may be empty ONLY for fold=lexicon_meta rows (not prompts);
    # v1 rows (minLength>=1) remain valid - this is an additive relaxation.
    props["prompt"] = {"type": "string", "description": "Empty string allowed for fold=lexicon_meta rows"}
    # new OPTIONAL properties (additive)
    new_props = {
        "family": {"type": "string", "enum": ["qwen3", "llama32", "gemma2", "qwen25"]},
        "tokenizer": {"type": "string", "minLength": 1},
        "variant_id": {"type": "string", "minLength": 1},
        "token_count": {"type": "integer", "minimum": 1},
        "token_ids": {"type": "array", "items": {"type": "integer"}},
        "align_shared_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "align_slot_start": {"type": "integer", "minimum": 0},
        "align_slot_end": {"type": "integer", "minimum": 0},
        "search_trail": {"type": "string", "minLength": 1},
        "decoded_start_tokens": {"type": "array", "items": {"type": "string"}},
        "blocked_reason": {"type": "string", "minLength": 1},
    }
    props.update(new_props)
    return s


@logger.catch(reraise=True)
def main() -> None:
    draft = json.loads((PROJ / "temp" / "pairs_v2_draft.json").read_text())
    lex = json.loads((PROJ / "temp" / "lexicon_meta_v2.json").read_text())["records"]
    acquired = json.loads((PROJ / "temp" / "tokenizers" / "acquired.json").read_text())

    # iteration-1 canonical strings (for variant tagging)
    i1 = json.loads((ITER1 / "pairs_manifest.json").read_text())
    canonical = {cat: (p["harmful"], p["benign"]) for cat, p in i1["pairs"].items()}

    rows: list[dict] = []
    for fam in FAMILIES:
        for cat in ORDER:
            rec = draft["families"][fam][cat]
            pnum = CAT2PAIR[cat]
            if not rec.get("count_equal"):
                # blocked category -> single placeholder row is NOT created; the
                # manifest/QC carry blocked_reason instead (plan STEP 9.3).
                logger.error(f"{fam}/{cat}: BLOCKED, no registry rows emitted")
                continue
            h, b = rec["harmful"], rec["benign"]
            is_canon = (h, b) == canonical[cat]
            if fam == "qwen3":
                variant = "iter1_verbatim"
                source = "template:screen_instrument:iter1_verbatim_qwen3"
            elif is_canon:
                variant = "canonical_reuse"
                source = f"template:screen_instrument_v2:{fam}:canonical_reuse"
            else:
                variant = "crafted"
                source = f"template:screen_instrument_v2:{fam}:crafted"
            pair_meta = {
                "family": fam, "tokenizer": EFFECTIVE_REPO[fam],
                "variant_id": f"{fam}|{pnum}|{variant}",
                "token_count": rec["harmful_tokens"],
                "token_ids": None,  # per member below
                "align_shared_fraction": rec.get("align_shared_fraction"),
                "align_slot_start": rec.get("align_slot_start"),
                "align_slot_end": rec.get("align_slot_end"),
                "search_trail": rec.get("search_trail"),
            }
            for typ, s, tid in (("harmful", h, rec["harmful_token_ids"]),
                                ("benign", b, rec["benign_token_ids"])):
                row = {
                    "id": f"sb_si2_{fam}_p{pnum[-2:]}_{typ[0]}",
                    "prompt": s,
                    "type": typ,
                    "category": cat if typ == "harmful" else "benign",
                    "fold": "screen_instrument",
                    "source": source,
                    "matched_pair_id": pnum,
                    "output": "",
                    "gold_label": None,
                    **pair_meta,
                    "token_ids": list(tid),
                }
                if rec.get("alignment_gate") == "count_equality_only":
                    row["notes"] = "alignment_gate=count_equality_only: " + rec.get(
                        "alignment_gate_note",
                        "count equality (hard gate) satisfied; alignment shared prefix/suffix "
                        "gate not fully met (documented fallback, plan STEP 9.3)")
                rows.append(row)

    # 4 lexicon_meta rows
    for fam in FAMILIES:
        lx = lex[fam]
        notes_parts = [lx["usage"], lx["caveat"]]
        if lx.get("qc_note"):
            notes_parts.append(lx["qc_note"])
        rows.append({
            "id": f"sb_lx_meta_{fam}",
            "prompt": "",
            "type": "lexicon_meta",
            "category": "lexicon",
            "fold": "lexicon_meta",
            "source": lx["source"],
            "matched_pair_id": "",
            "output": "",
            "gold_label": None,
            "family": fam,
            "tokenizer": lx["tokenizer"],
            "token_ids": lx["token_ids"],
            "decoded_start_tokens": lx["decoded_start_tokens"],
            "notes": " | ".join(notes_parts),
        })

    rows.sort(key=lambda r: r["id"])
    (PROJ / "data_out.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    logger.info(f"wrote data_out.json with {len(rows)} rows "
                f"({sum(1 for r in rows if r['fold']=='screen_instrument')} screen + "
                f"{sum(1 for r in rows if r['fold']=='lexicon_meta')} lexicon_meta)")

    # schema v2
    schema_v2 = build_schema_v2()
    (PROJ / "schema.json").write_text(json.dumps(schema_v2, indent=2, ensure_ascii=False))
    logger.info("wrote schema.json (v2, additive)")

    # full_data_out.json (exp_sel_data_out view)
    groups = []
    fold_int = {"screen_instrument": 0, "lexicon_meta": 4}
    for fam in FAMILIES:
        fam_rows = [r for r in rows if r["fold"] == "screen_instrument" and r["family"] == fam]
        examples = []
        for i, r in enumerate(sorted(fam_rows, key=lambda x: x["id"])):
            opp = "benign" if r["type"] == "harmful" else "harmful"
            counterpart = next((c for c in fam_rows
                                if c["matched_pair_id"] == r["matched_pair_id"] and c["type"] == opp), None)
            ex = {
                "input": r["prompt"],
                "output": counterpart["prompt"] if counterpart else "",
                "metadata_row_index": i,
                "metadata_fold": fold_int["screen_instrument"],
                "metadata_feature_names": ["prompt_tokens"],
                "metadata_prompt_type": r["type"],
                "metadata_category": r["category"],
                "metadata_source": r["source"],
                "metadata_matched_pair_id": r["matched_pair_id"],
                "metadata_task_type": "prompt_registry",
                "metadata_family": r["family"],
                "metadata_tokenizer": r["tokenizer"],
                "metadata_token_count": r["token_count"],
                "metadata_token_ids": r["token_ids"],
            }
            if r.get("notes"):
                ex["metadata_notes"] = r["notes"]
            examples.append(ex)
        groups.append({"dataset": f"refusal_boundary_screen_instrument_{fam}",
                       "examples": examples})
    lx_examples = []
    for i, r in enumerate(sorted((x for x in rows if x["fold"] == "lexicon_meta"), key=lambda x: x["id"])):
        lx_examples.append({
            "input": "",
            "output": "",
            "metadata_row_index": i,
            "metadata_fold": fold_int["lexicon_meta"],
            "metadata_feature_names": [],
            "metadata_prompt_type": "lexicon_meta",
            "metadata_category": "lexicon",
            "metadata_source": r["source"],
            "metadata_matched_pair_id": "",
            "metadata_task_type": "prompt_registry",
            "metadata_family": r["family"],
            "metadata_tokenizer": r["tokenizer"],
            "metadata_token_ids": r["token_ids"],
            "metadata_decoded_start_tokens": r["decoded_start_tokens"],
            "metadata_notes": r["notes"],
        })
    groups.append({"dataset": "refusal_boundary_lexicon_meta", "examples": lx_examples})
    full = {
        "metadata": {
            "source": "refusal-boundary dataset artifact v2 (gen_plan_dataset_1_idx1, iter_2)",
            "description": (
                "Tokenizer-agnostic screen instrument: 6 matched harmful/benign pairs x 4 "
                "tokenizer families (qwen3/llama32/gemma2/qwen25) with VERIFIED identical "
                "token counts under each family tokenizer, plus per-family Arditi Table 4 "
                "refusal start-token lexicon metadata. INPUT=prompt; OUTPUT=matched pair "
                "counterpart (screen) / '' (lexicon_meta)."
            ),
            "reservation_rule": (
                "Iteration-1 fold=confirm rows (60 harmful + 40 benign) remain reserved and "
                "are untouched by this artifact; the v2 screen strings are canonical stimuli "
                "for the iteration-2 experiment and must be embedded byte-identically."
            ),
        },
        "datasets": groups,
    }
    (PROJ / "full_data_out.json").write_text(json.dumps(full, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"wrote full_data_out.json with {sum(len(g['examples']) for g in groups)} examples "
                f"across {len(groups)} groups")

    # pairs_manifest_v2.json
    # normalize cross_family_reuse draft into family -> {category: [others]}
    reuse: dict[str, dict[str, list[str]]] = {fam: {} for fam in FAMILIES}
    for fam, lst in draft.get("cross_family_reuse", {}).items():
        for item in lst:
            for cat, other in item.items():
                reuse.setdefault(fam, {}).setdefault(cat, []).append(other)
    manifest = {
        "schema_version": "2.0",
        "matching_protocol": (
            "identical syntactic skeleton per pair; only the intent slot differs; identical "
            "ASCII punctuation and case; VERIFIED identical token counts (hard gate) in [8,25] "
            "under the family's own tokenizer WITHOUT special tokens (iteration-1 convention); "
            "registered byte-exact (json.dump ensure_ascii=False); never re-normalized"
        ),
        "families": {fam: {
            "family": fam,
            "effective_tokenizer_repo": EFFECTIVE_REPO[fam],
            "mirror_fallback_used": False,
            "gating_note": acquired["gating_table_note"] if "gating_table_note" in acquired else
                           f"see temp/tokenizers/acquired.json for gating probe results",
            "pairs": {cat: {
                "pair": CAT2PAIR[cat],
                "category": cat,
                "harmful": draft["families"][fam][cat]["harmful"],
                "benign": draft["families"][fam][cat]["benign"],
                "harmful_chars": draft["families"][fam][cat]["harmful_chars"],
                "benign_chars": draft["families"][fam][cat]["benign_chars"],
                "harmful_tokens": draft["families"][fam][cat]["harmful_tokens"],
                "benign_tokens": draft["families"][fam][cat]["benign_tokens"],
                "count_equal": draft["families"][fam][cat]["count_equal"],
                "harmful_token_ids": draft["families"][fam][cat]["harmful_token_ids"],
                "benign_token_ids": draft["families"][fam][cat]["benign_token_ids"],
                "align_shared_fraction": draft["families"][fam][cat].get("align_shared_fraction"),
                "align_slot_start": draft["families"][fam][cat].get("align_slot_start"),
                "align_slot_end": draft["families"][fam][cat].get("align_slot_end"),
                "align_prefix_shared": draft["families"][fam][cat].get("align_prefix_shared"),
                "align_suffix_shared": draft["families"][fam][cat].get("align_suffix_shared"),
                "alignment_gate": draft["families"][fam][cat].get("alignment_gate"),
                "alignment_gate_note": draft["families"][fam][cat].get("alignment_gate_note"),
                "search_trail": draft["families"][fam][cat].get("search_trail"),
                "combos_total": draft["families"][fam][cat].get("combos_total"),
                "candidate_idx": draft["families"][fam][cat].get("candidate_idx"),
                "template": draft["families"][fam][cat].get("template"),
                "iter1_counts_match": draft["families"][fam][cat].get("iter1_counts_match"),
                "cross_family_reuse": sorted(reuse.get(fam, {}).get(cat, [])),
                "blocked_reason": draft["families"][fam][cat].get("blocked_reason"),
            } for cat in ORDER},
        } for fam in FAMILIES},
        "cross_family_reuse": reuse,
        "lexicon_meta": {fam: {"token_ids": lex[fam]["token_ids"],
                               "decoded_start_tokens": lex[fam]["decoded_start_tokens"],
                               "source": lex[fam]["source"]} for fam in FAMILIES},
    }
    (PROJ / "pairs_manifest_v2.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info("wrote pairs_manifest_v2.json")


if __name__ == "__main__":
    main()