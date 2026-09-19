import os, sys, json, traceback
sys.path.insert(0, "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1")
os.environ["HF_HOME"] = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/cache/hf_home"
import instrument as instr
from method import Metrics, jackknife_scores, CANDIDATES, BASELINES

instrument = instr.build_instrument()
ev = json.load(open("/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/results/evidence/0.6B__tuned__Qwen3-0.6B.json"))
m = Metrics(ev, lexicon=instrument.lexicon)
try:
    mm = m.compute_all()
    print("METRICS OK", mm["scores"])
except Exception:
    traceback.print_exc()
try:
    for ex in list(ev["pairs"].keys()):
        jk = jackknife_scores(ev, instrument.lexicon, ex)
        print("JK", ex, jk)
except Exception:
    traceback.print_exc()