#!/usr/bin/env python3
"""data.py - Standardize the refusal-boundary registry into
exp_sel_data_out.json (full_data_out.json).

Groups (the artifact plan's target_num_datasets=4):
  1. refusal_boundary_screen_instrument (12 rows; 6 token-equal pairs)
  2. refusal_boundary_confirm_harmful   (60 rows; 10 x 6 categories)
  3. refusal_boundary_confirm_benign    (40 rows)
  4. refusal_boundary_lexicon_validation(20 rows; 8-token snippets + gold labels)

ONE example per row (132 examples total). Per-example fields:
  input  = prompt (byte-exact)
  output = screen_instrument: the matched pair counterpart (byte-exact)
           confirm rows: "" (no target; stimulus-only)
           lexicon rows: greedy 8-token output snippet
  metadata_* flat fields (no nested objects), as required by the schema.
"""
from pathlib import Path
import json

from loguru import logger

logger.remove()
logger.add(lambda msg: print(str(msg).strip()), level="INFO")


def main() -> None:
    rows = json.loads(Path("data_out.json").read_text())
    logger.info(f"loaded {len(rows)} registry rows from data_out.json")

    groups = {
        "refusal_boundary_screen_instrument": [
            r for r in rows if r["fold"] == "screen_instrument"
        ],
        "refusal_boundary_confirm_harmful": [
            r for r in rows if r["fold"] == "confirm" and r["type"] == "harmful"
        ],
        "refusal_boundary_confirm_benign": [
            r for r in rows if r["fold"] == "confirm" and r["type"] == "benign"
        ],
        "refusal_boundary_lexicon_validation": [
            r for r in rows if r["fold"] == "lexicon_validation"
        ],
    }
    fold_int = {"screen_instrument": 0, "confirm": 1, "lexicon_validation": 3}
    fold_int_benign = 2

    datasets = []
    total = 0
    for name, group_rows in groups.items():
        examples = []
        for i, r in enumerate(sorted(group_rows, key=lambda x: x["id"])):
            if r["fold"] == "screen_instrument":
                # output = the matched counterpart prompt (byte-exact)
                opposite = "benign" if r["type"] == "harmful" else "harmful"
                counterpart = None
                for cand in group_rows:  # need same pair id + opposite type
                    if cand["matched_pair_id"] == r["matched_pair_id"] and cand["type"] == opposite:
                        counterpart = cand
                        break
                output = counterpart["prompt"] if counterpart else ""
                fold = fold_int["screen_instrument"]
            elif r["fold"] == "confirm":
                output = ""
                fold = fold_int["confirm"] if r["type"] == "harmful" else fold_int_benign
            else:
                output = r["output"]
                fold = fold_int["lexicon_validation"]
            ex = {
                "input": r["prompt"],
                "output": output,
                "metadata_row_index": i,
                "metadata_fold": fold,
                "metadata_feature_names": ["prompt_tokens"],
                "metadata_prompt_type": r["type"],
                "metadata_category": r["category"],
                "metadata_source": r["source"],
                "metadata_matched_pair_id": r["matched_pair_id"],
                "metadata_task_type": "prompt_registry",
            }
            if r["fold"] == "lexicon_validation":
                ex["metadata_gold_label"] = r["gold_label"]
            if "notes" in r:
                ex["metadata_notes"] = r["notes"]
            examples.append(ex)
        datasets.append({"dataset": name, "examples": examples})
        total += len(examples)
        logger.info(f"{name}: {len(examples)} examples")

    out = {
        "metadata": {
            "source": "refusal-boundary dataset artifact (gen_plan_dataset_1_idx2)",
            "description": (
                "Refusal-boundary test prompts and lexicon: screen instrument (6 matched "
                "harmful/benign pairs, token-equal via Qwen3 tokenizer), reserved confirm "
                "corpus (60 harmful 10x6 categories + 40 benign), and 20 lexicon-validation "
                "gold items (greedy 8-token snippets + refusal/compliance labels). "
                "INPUT=prompt; OUTPUT=matched pair counterpart (screen_instrument) / '' "
                "(confirm) / snippet (lexicon_validation)."
            ),
            "tokenizer": "Qwen/Qwen3-0.6B",
            "reservation_rule": (
                "fold=confirm rows are reserved; they are used ONLY by iteration-2 "
                "confirmation/evaluation artifacts and must NEVER be seen by the iteration-1 "
                "screen (including the same-iteration experiment). fold=screen_instrument rows "
                "are canonical stimulus strings; the same-iteration experiment embeds "
                "byte-identical copies."
            ),
        },
        "datasets": datasets,
    }
    Path("full_data_out.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    logger.info(f"wrote full_data_out.json with {total} examples across {len(datasets)} datasets")


if __name__ == "__main__":
    main()