import os, torch
os.environ["HF_HOME"] = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/cache/hf_home"
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
mod = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)
mod.to("cuda"); mod.eval()

manifest = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_dataset_1/pairs_manifest.json"
import json, sys
sys.path.insert(0, "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
from instrument import REFUSAL_LEXICON_SEED

def lexicon_hit(text):
    low = text.lower()
    return any(ph in low for ph in REFUSAL_LEXICON_SEED)

prompts = []
for cat, e in json.load(open(manifest))["pairs"].items():
    prompts.append(("H", cat, e["harmful"]))
    prompts.append(("B", cat, e["benign"]))
for tag, cat, p in prompts:
    if repo.endswith("Base"):
        s1 = p
        mode = "raw (base)"
    else:
        try:
            s1 = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False, tokenize=False)
            mode = "enable_thinking=False OK"
        except Exception as e:
            s1 = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
            mode = f"kwarg failed: {type(e).__name__}"
    ids = tok.encode(s1, add_special_tokens=True, return_tensors="pt").to("cuda")
    gen = mod.generate(ids, max_new_tokens=8, do_sample=False, temperature=1.0, top_p=1.0)
    out = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
    print(f"[{mode}] {p!r:45} -> {out!r}")