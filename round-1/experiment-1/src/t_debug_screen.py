import os, sys, traceback
os.environ["HF_HOME"] = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/cache/hf_home"
sys.path.insert(0, "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
import torch
from method import ModelAnalyzer, ModelSpec, apply_memory_limits
import instrument as instr

apply_memory_limits(30.0, vram_fraction=0.85)
device = "cuda"
instrument = instr.build_instrument()
spec = ModelSpec(family="0.6B", role="tuned", repo_id="Qwen/Qwen3-0.6B", is_chat=True, gated=False)
ana = ModelAnalyzer(spec=spec, instrument=instrument, out_dir=os.environ.get("OUT", "."),
                    device=device, num_alphas=3, splice_layers=2, enable_c4=True)
ana.load()
import torch
_orig_interp = ana._interp_forward
def _dbg_interp(emb_b, emb_h):
    print("DBG emb_b.requires_grad", emb_b.requires_grad, "emb_h", emb_h.requires_grad)
    out = _orig_interp(emb_b, emb_h)
    hs = out.hidden_states
    print("DBG hs[0].requires_grad", hs[0].requires_grad, "hs[3]", hs[3].requires_grad,
          "logits", out.logits.requires_grad, "grad_enabled", torch.is_grad_enabled())
    return out
ana._interp_forward = _dbg_interp
cats = list(instrument.pairs.keys())[:2]
pool = {c: list(dict.fromkeys([instrument.pairs[c]] + instr.PAIR_POOL[c])) for c in instrument.pairs}
try:
    ev = ana.screen(cats, pool)
    print("SCREEN OK cats:", list(ev["pairs"].keys()), "splice_max:", ev["splice_max_abs_diff"])
    m = getattr(__import__("method"), "Metrics")(ev, lexicon=instrument.lexicon)
    mm = m.compute_all()
    print("METRICS OK", {k: v if isinstance(v, float) else v.get("value") if isinstance(v, dict) else None for k, v in mm["scores"].items()})
except Exception:
    traceback.print_exc()