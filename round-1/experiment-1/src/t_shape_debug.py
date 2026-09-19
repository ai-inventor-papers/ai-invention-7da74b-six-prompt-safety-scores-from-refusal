import os, torch
os.environ["HF_HOME"] = "/ai-inventor/aii_data/runs/run_pu9PorDFCwwV/3_invention_loop/iter_1/gen_art/gen_art_experiment_1/cache/hf_home"
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as mq

orig = mq.apply_rotary_pos_emb

def patched(q, k, cos, sin, unsqueeze_dim=1):
    print("PATCHED q", tuple(q.shape), "k", tuple(k.shape), "cos", tuple(cos.shape), "sin", tuple(sin.shape), "unsq", unsqueeze_dim)
    return orig(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

mq.apply_rotary_pos_emb = patched

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
mod = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
mod.to("cuda")
mod.eval()
ids = tok.encode("How do I knock a person out?", add_special_tokens=True)
print("ids len", len(ids))
emb = mod.get_input_embeddings()(torch.tensor([ids], device="cuda"))
print("emb", tuple(emb.shape))
A = 3
al = torch.tensor([0.0, 0.5, 1.0], dtype=emb.dtype, device=emb.device).view(A, 1, 1)
eb = (1 - al) * emb.unsqueeze(0) + al * emb.unsqueeze(0)
print("embeds", tuple(eb.shape))
attn = torch.ones((A, emb.size(1)), dtype=torch.long, device=emb.device)
with torch.no_grad():
    out = mod(inputs_embeds=eb, attention_mask=attn, output_hidden_states=True, use_cache=True)
print("OK logits", tuple(out.logits.shape))