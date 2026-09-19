# Grounding the refusal-boundary safety metric

## Summary

Web research (run 2026-09-19) grounding an iteration-2 experiment on a refusal-boundary safety metric: (1) exact HF model IDs/gating/dtypes/sizes verified via the HF REST + tree APIs for the Qwen3-4B/0.6B, Llama-3.2-1B and Gemma-2-2B trios plus community abliterated variants - three plan-guessed IDs were wrong and corrected (Qwen3-0.6B-Instruct-2507 does not exist; huihui-ai ID is Huihui-Qwen3-4B-Instruct-2507-abliterated; a huihui 0.6B abliteration exists); (2) published safety anchors are sparse exactly as hypothesized (Abliterlitics HarmBench numbers for the 4B trio, Gemma-2-IT-2B human-safety win rate from the Gemma 2 report; NOT FOUND for Llama-3.2-1B and Qwen3-0.6B so the internal 60/40 eval is the anchor); (3) exact abliteration recipe extracted from Arditi et al. 2406.11717 (Eq. 3 / Eq. 17, Table 4 token sets, Table 5 layer selection) and the FailSpy/huihui-ai abliterator.py weight-edit code (W <- W - einsum(W,r_hat)*r_hat over layers 1..n-1, TOXIC/alpaca data); (4) dated novelty saturation search: the exact lane (input-embedding interpolation psychometric curve r(alpha) scored by boundary sharpness/locus as a safety metric) is NOT occupied; nearest lanes mapped with axis/readout/purpose differences (LFJ attack 2508.10029, Geometry-Lite 2605.20241, SafeVec/RAS 2606.25750, RASS 2505.18325, DPS 2510.03271, Semantic Confusion 2512.01037); 13/13 candidate arXiv IDs verified FOUND; (5) refusal readout protocol: family-specific start-token sets + OR-Bench starts-with keywords + XSTest 3-way taxonomy + greedy-determinism caveats; (6) knowledge-action-gap evidence secured: 2603.18353 (98.2% AUROC vs 45.1% sensitivity, 53pp gap, SAE steering zero effect), 2507.11878 (harmfulness/refusall separate directions, survives adversarial fine-tuning), 2512.12066 (18-28% seed flips), 2607.13346 (probes flag but cannot steer). Deliverables: research_out.json + research_report.md with feasibility table, safety-anchor table, abliteration recipe, novelty memo, readout protocol, gap citations, risk register.

## Research Findings

**Model availability (Phase 1, verified 2026-09-19 via the HuggingFace REST + tree APIs).** All ten planned models exist; three plan-guessed IDs were wrong. Qwen/Qwen3-4B [1] and Qwen/Qwen3-4B-Instruct-2507 [2] exist ungated (Apache-2.0) at ~8.04 GB bf16 (3 safetensors shards totaling 8,044,982,000 bytes each). Qwen/Qwen3-0.6B [3] exists ungated (Apache-2.0, 22.5M downloads), has a chat template, and IS the 0.6B instruct model (its tags show base_model Qwen3-0.6B-Base); the plan's 'Qwen3-0.6B-Instruct-2507' does not exist (persistent HTTP 401 from API and page, absent from every search - the Qwen3 2507 refresh skipped the 0.6B size). The plan's 'huihui-ai/Qwen3-4B-Abliterated-2507' also does not exist; the real community abliteration of Qwen3-4B-Instruct-2507 is huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated (~8.04 GB bf16; gated="auto", a free contact-info gate, not a license gate) [5]. A community abliterated 0.6B DOES exist - huihui-ai/Qwen3-0.6B-abliterated (gated="auto", base=Qwen/Qwen3-0.6B, 1.19 GB single shard vs the official 1.50 GB; iteration 2 must check the config.json torch_dtype for this discrepancy) [4]. meta-llama/Llama-3.2-1B [6] and -1B-Instruct [7] exist but are gated="manual" (Llama 3.2 license acceptance + HF token) at ~2.47 GB bf16; google/gemma-2-2b [8] and gemma-2-2b-it [9] are gated="manual" (Gemma terms + HF token), the base stored FP32 at ~10.46 GB (5.2 GB in fp16) and the -it bf16 at ~5.23 GB. Community abliterated alternates exist for Llama-3.2-1B-Instruct (huihui-ai/Llama-3.2-1B-Instruct-abliterated [10]) and Gemma-2-2B-it (IlyaGusev/gemma-2-2b-it-abliterated [34]). Feasibility verdict: every planned model fits a ~14 GB CPU-fp16 container (largest 8.04 GB); iteration 2 must set HF_TOKEN with accepted Llama/Gemma licenses or drop those six rows.

**Published safety anchors (Phase 2).** Exact-model numbers are sparse exactly as hypothesized. The independent forensics site Abliterlitics [11] gives the Qwen3-4B-Instruct-2507 trio (HarmBench, 400 behaviours, max_tokens=2048, temperature=0): base ASR 24.8% (301/400 refusals) and MMLU 70.60; huihui abliterated ASR 95.5% (18/400 refusals remain), MMLU 69.34, KL(batchmean) 0.309; Heretic ASR 99.2%, HauhauCS ASR 100.0% with KL 0.161; TruthfulQA drops 7-9 points across techniques; weight forensics show the huihui 4B edit touched 36/36 layers (o_proj, down_proj, gate_proj, v_proj; 2.13% relative magnitude) and refusals were regex-classified with claimed 100% coverage. Proxy controls: Qwen3-8B refuses 84.7% of AdvBench but complies with 98.3% of novel prompt families (83-pp gap; chi2=80.5, p<1e-18, Cramer's V 0.82) [12]; Qwen3-8B sits inside the 18-28% decision-flip band of the Instability-of-Safety sweep (876 prompts x 20 sampling configs = 70,080 responses; Llama 3.1 8B single-shot evaluation agrees with majority ground truth only 92.5% on average - 98.7% greedy down to 90.3% at t=1.0 - and flip rate rises from 5.1% at t=0 to 23.6% at t=1.0) [13]. For gemma-2-2b-it, Gemma 2 report Table 15 (arXiv 2408.00118) gives a human side-by-side safety win rate vs GPT-4o of 57.5% (W/T/L 53/9/38) plus the qualitative claim that even small Gemma 2 models are safer than GPT-4o on the held-out safety set [14]. NOT FOUND: any public refusal/harmlessness number for Llama-3.2-1B(-Instruct) or Qwen3-0.6B; the Qwen3 technical report PDF (2505.09388, 116k chars) contains no safety/HarmBench/refusal evaluation text at all [35]. Consequence: the internal held-out 60-harmful/40-benign lexicon evaluation is the primary anchor; the numbers above are cross-checks where they exist.

**Canonical abliteration recipe (Phase 3).** Arditi et al. (arXiv 2406.11717) [15]: the refusal direction is the difference-in-means of activations between harmful and harmless chat-formatted prompts, computed per layer and per post-instruction token position, with the last token position (i*=-1) winning the bypass/induce/KL-score selection (e.g., QWEN 1.8B selects layer 15/24, QWEN 7B layer 17/32, Table 5). Activation addition is x(l)' <- x(l) + r(l) (Eq. 3, quoted); removal is x(l)' <- x(l) - r(l) (Eq. 17, quoted); the weight-edit version is the orthogonal projection (I - r_hat r_hat^T) applied to the residual-stream write paths. Table 4 gives per-family refusal start-token sets: QWEN {40,2121} ("I'm sorry", "As an AI"), GEMMA {235285} ("I cannot"), LLAMA-3 {40} ("I cannot"); Eq. 6 defines P_refusal as the sum of next-token probabilities over the set R. The FailSpy/abliterator tool (the huihui-ai lineage) [16] implements, per layer in range(1, n_layers), for attention-output and MLP-output matrices: proj = einsum(W, r_hat) * r_hat; W <- W - proj - exactly (I - r_hat r_hat^T)W - using transformer_lens in bf16, harmful data from Undi95/orthogonal-activation-steering-TOXIC, harmless data from tatsu-lab/alpaca, and the Llama-3 refusal token sets {4250,14931,89735,20451,11660,11458,956} (negative) and {32,1271,8586,96556,78145} (positive). Arditi's demo pipeline (andyrdt/refusal_direction) [17] and the pure-HF re-implementation (sumandora/remove-refusals-with-transformers) [18] are available references. Recommended start for Qwen3-0.6B (24 layers): l* ~13-15 by scaling Qwen7B's 17/32, plus an all-layers variant, with l* re-derived from the bypass/induce/KL selection rather than copied.

**Novelty saturation search (Phase 4, dated 2026-09-19).** 16 queries across scholarly (OpenAlex) and general (Exa/marginalia) indexes; 13/13 candidate arXiv IDs verified FOUND (2406.11717, 2502.17420, 2506.16078, 2507.11878, 2510.02768, 2512.01037, 2512.12066, 2602.02132, 2606.16349, 2606.25750, 2608.07528, 2608.17843, 2608.25390), no 404s, no invented titles (full list in research_report.md). VERDICT: the exact lane - interpolating input embeddings across a matched harmful<->benign prompt pair, reading the greedy-decoded refusal-vs-compliance transition r(alpha), and scoring boundary sharpness x full-range transition as a safety metric - is NOT occupied. Nearest occupied lanes, all differentiated: Latent Fusion Jailbreak (2508.10029) [19] pairs a harmful query with a structurally similar benign counterpart and interpolates hidden states at selected layers/token positions with gradient-optimized mixing - but it is an ATTACK (ASR 94.13% white-box, judge readout, rejection sampling; no input-level alpha sweep, no curve scoring); SafeVec/RAS (2606.25750) [23] - the plan's designated direction-alignment baseline - extracts reference-model refusal directions and scores target models by cosine alignment of hidden states (calibrated 0-100 RAS; validated on Llama/Gemma/Qwen including abliterated variants); Geometry-Lite (2605.20241) [20] probes layer-wise signed margins from prompt-token representations (detection probe, 1.2B-70B backbones); Decision Potential Surface (2510.03271) [21] approximates the input-space decision boundary for analysis; the EMNLP-2025 RASS overrefusal paper (2505.18325) [22] and the companion representation-intervention work (2511.19009) [36] probe or steer the safety decision boundary to generate/repair near-boundary benign prompts for overrefusal mitigation, not for scoring; Semantic Confusion (2512.01037) [25] measures refusal inconsistency across meaning-preserving paraphrases (Confusion Index/Rate/Depth); latent-perturbation robustness (2506.16078) [26] reads NLL robustness under arbitrary latent shifts as a diagnostic; adjacent activation-level work - Safety Arithmetic (2406.11801, steering to align behavior), Step-Wise Refusal Dynamics [37], Latent-space Refusal Evasion [38], Refusal Before Decoding [39] - either adds directions to change behavior or reads activations, whereas the metric only READS a behavioral transition curve over an input-level interpolation; embedding-perturbation jailbreaks (Adaptive Prompt Embedding Optimization 2604.24983; Universal Magic Words 2501.18280) are adversarial attacks, excluded from the metric lane by construction (their readout is attack success, not boundary scoring). Claim novelty as a NEW metric, and name RAS/SafeVec, probes, Geometry-Lite margins and LFJ (as attack control) as the comparison baselines; the memo is dated and limited to the indexes searched.

**Refusal readout protocol (Phase 5).** No-judge classification is standard: OR-Bench's official alignment checker [32] uses starts-with keywords ['I apologize', 'I'm sorry', 'I cannot', 'I can't']; XSTest (2308.01263) [33] uses a three-way Full Compliance / Full Refusal / Partial Refusal taxonomy plus a string-match detector ('starts with "I'm sorry", "as an AI", "I cannot", or variations'), counting alternatives-without-answer as refusal; Arditi's per-family start-token sets and P_refusal over next-token probabilities give the token-level primitive [15]; FailSpy's token sets exist precisely because BPE splits phrases ('can't' vs 'can not') [16]. Recommended protocol: greedy decode (do_sample=False, temperature=0.0), max_new_tokens 8-16, lexicon match on the first ~2 tokens of the prefix, empty/short no-phrase output = compliance, validate on ~20 hand-labeled outputs per model, re-run each (model, alpha) at 2-3 seeds, pin eager attention + bf16, continuous fallback = Arditi P_refusal [15,16,32,33,11].

**Knowledge-action gap (Phase 6).** The motivation is now fully evidenced. arXiv 2603.18353 (Basu et al., Mar 2026; 400 physician-adjudicated clinical triage vignettes; Qwen2.5-7B-Instruct + Steerling-8B): linear probes reach 98.2% AUROC while output sensitivity is only 45.1% - a 53-percentage-point knowledge-action gap; SAE feature steering produced zero effect despite 3,695 significant features; concept-bottleneck steering corrected 20% of missed hazards but disrupted 53% of correct detections (p=0.84) - scope caveat: single clinical domain [27]. arXiv 2507.11878 'LLMs Encode Harmfulness and Refusal Separately' (Zhao, Huang, Wu, Bau, Shi): the harmfulness direction is distinct from the refusal direction; steering harmfulness makes benign prompts look harmful; adversarial fine-tuning that makes models accept harmful instructions leaves the internal harmfulness belief nearly intact (Latent Guard) [29]. arXiv 2512.12066: refusal decisions flip across seeds/temperatures for 18-28% of prompts, justifying deterministic greedy readouts [13]. arXiv 2607.13346 'The Refusal Residue' (Mehta, ICML 2026 MI Workshop): probes can flag alignment faking (AUROC 0.87 on Llama-3.1-8B under leave-one-query-out) but the detected direction does not control behavior (steering |h|<0.08 over 2,000 runs) and naive probes overstate detectability by 0.2-0.3 AUROC [28]. Titles verified for 2608.07528 Knowing-Saying Gap [30] and 2608.17843 Encoded but Not Actionable [31] (detail not yet extracted; use with domain caveats). Framing: knowledge-based scores over-score models that still know harmfulness but stopped acting on it; the boundary metric measures the routing action - the transition curve of outputs under interpolation - so it cannot be gamed by knowledge that never reaches behavior; refusal-geometry work independently shows the post-hoc safety update is 'thin but sharp' and nearly orthogonal to capabilities [24], consistent with a separable routing gate.

**Confidence and caveats.** High for everything API/arXiv-verified (IDs, gating, sizes, title, anchor numbers); medium for Ablierlitics (community forensics, single reporter; cross-check ASR 24.8/95.5 before citing as ground truth); medium-low for the huihui-0.6B dtype discrepancy (1.19 vs 1.50 GB). The novelty verdict is dated 2026-09-19 and bound to the indexes searched; re-run the LFJ/Geometry-Lite lanes before submission. Explicit NOT FOUND entries: public anchors for Llama-3.2-1B(-Instruct) and Qwen3-0.6B; the model IDs 'Qwen3-0.6B-Instruct-2507' and 'huihui-ai/Qwen3-4B-Abliterated-2507'. Per-phase recommendations: (1) use the corrected IDs and set HF_TOKEN or drop the Llama/Gemma rows; (2) anchor the 4B row to Ablierlitics, internal eval is the anchor for 0.6B/1B; (3) self-abliterate 0.6B with the FailSpy formula at l* ~13-15 plus an all-layers variant; (4) claim novelty with the dated memo and RAS/LFJ/Geometry-Lite baselines; (5) adopt the greedy + lexicon protocol; (6) cite [27,29,13,28] for the motivation paragraph.

## Sources

[1] [HF API metadata: Qwen/Qwen3-4B](https://huggingface.co/api/models/Qwen/Qwen3-4B) (2025) — Verified id, gated=false, apache-2.0, 7,242,404 downloads, lastModified 2025-07-26; tree API -> 3 safetensors shards totaling 8,044,982,000 B (~8.04 GB bf16)

> "gated":false

Locator: HF REST API

[2] [HF API metadata: Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507) (2025) — Verified id, gated=false, apache-2.0, 3,992,032 downloads, lastModified 2025-09-17; tree API -> same ~8.04 GB bf16 layout

[3] [HF API metadata: Qwen/Qwen3-0.6B](https://huggingface.co/api/models/Qwen/Qwen3-0.6B) (2025) — Verified id, gated=false, apache-2.0, 22,498,727 downloads; has chat template and tags base_model:Qwen/Qwen3-0.6B-Base -> this is the 0.6B instruct; model.safetensors = 1,503,300,328 B (~1.50 GB)

[4] [HF API metadata: huihui-ai/Qwen3-0.6B-abliterated](https://huggingface.co/api/models/huihui-ai/Qwen3-0.6B-abliterated) (2025) — Verified id, gated="auto" (contact-info gate), apache-2.0, base_model=Qwen/Qwen3-0.6B, 45 downloads, lastModified 2025-06-19; model.safetensors = 1,192,135,096 B (~1.19 GB) vs official 1.50 GB (check dtype in iteration 2)

> "gated":"auto"

Locator: HF REST API

> "downloads":45

Locator: HF REST API

[5] [HF API tree: huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated](https://huggingface.co/api/models/huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated/tree/main) (2025) — Verified real community ID for the Qwen3-4B-Instruct-2507 abliteration (plan-guessed ID 404s); 2 shards 4,967,215,360 + 3,077,766,632 B = ~8.04 GB bf16; ID confirmed independently via search + Abliterlitics

[6] [HF API metadata: meta-llama/Llama-3.2-1B](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B) (2024) — Verified id, gated="manual", license llama3.2, 902,820 downloads; model.safetensors = 2,471,645,608 B (~2.47 GB); HF token + license acceptance required

> "gated":"manual"

Locator: HF REST API

[7] [HF API metadata: meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/api/models/meta-llama/Llama-3.2-1B-Instruct) (2024) — Verified id, gated="manual", llama3.2, 6,921,311 downloads, lastModified 2024-10-24; model.safetensors = 2,471,645,608 B

[8] [HF API metadata: google/gemma-2-2b](https://huggingface.co/api/models/google/gemma-2-2b) (2024) — Verified id, gated="manual", 170,008 downloads; 3 shards = 10,457,400,944 B (~10.46 GB) => base repo stored FP32; cast to fp16 (~5.2 GB) for CPU

> "gated":"manual"

Locator: HF REST API

[9] [HF API metadata: google/gemma-2-2b-it](https://huggingface.co/api/models/google/gemma-2-2b-it) (2024) — Verified id, gated="manual", 672,553 downloads; 2 shards 4,988,025,760+240,691,728 B = ~5.23 GB bf16

[10] [huihui-ai/Llama-3.2-1B-Instruct-abliterated (search snippet)](https://huggingface.co/huihui-ai/Llama-3.2-1B-Instruct-abliterated) (2025) — Community abliterated variant of Llama-3.2-1B-Instruct exists (top search hit); alternates nztinversive/llama3.2-1b-Uncensored, Elstuhn/..., cazzz307/... 'Llama-3.2-1B instruct model abliterated and uncensored at semi-deep layer'

[11] [Abliterlitics: Qwen3-4B-Instruct-2507 Abliteration Comparison](https://abliterlitics.dev/models/qwen3-4b/) (dreamfast (Abliterlitics); 2026) — EXACT safety anchors for the 4B trio: HarmBench 400 (base 301/400 refusals ASR 24.8%; Huihui 18/400 ASR 95.5%; Heretic 3/400 ASR 99.2%; HauhauCS 0/400 ASR 100%); MMLU 70.60/69.34; KL batchmean 0.161-0.310; weight forensics: Huihui edits 36/36 layers o_proj+down_proj+gate_proj+v_proj at 2.13% magnitude; 'Refusals were classified via regex matching, which captured 100% of refusals cleanly'

> 301/400| 24.8%

Locator: Safety: HarmBench table / intro

> 18/400| 95.5%

Locator: Safety: HarmBench table / intro

> Huihui struggles on this model. It only reaches 95.5% safety removal, leaving 18 harmful requests still refused.

Locator: Safety: HarmBench table / intro

> Refusals were classified via regex matching, which captured 100% of refusals cleanly

Locator: Safety: HarmBench table / intro

[12] [FailureFirst Report 229: Qwen3 Benchmark Overfitting Analysis (2026-03-24)](https://failurefirst.org/research/reports/229-qwen3-overfitting-analysis/) (FailureFirst; 2026) — PROXY anchor for Qwen3 family: Qwen3-8B refuses 84.7% of AdvBench prompts but complies with 98.3% of novel-family prompts (83-pp gap; chi-square 80.5, p<1e-18, Cramer's V 0.82)

> refuses 84.7% of AdvBench

Locator: Executive Summary

[13] [The Instability of Safety: How Random Seeds and Temperature Expose Inconsistent LLM Refusal Behavior (Erik Larsen)](https://arxiv.org/abs/2512.12066) (Erik Larsen; 2025) — 4 models (Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B), 876 prompts, 20 configs, 70,080 responses; 18-28% of prompts exhibit decision flips; Llama 3.1 8B: 32% flip rate (LLama-70B judge), single-shot agrees with ground truth 92.5% avg (98.7% greedy, 90.3% t=1.0); flips 5.1% at t=0 -> 23.6% at t=1.0; 14.3% substantial instability (SSI<0.8)

> 28% of prompts exhibit decision flips

Locator: Abstract

> single-shot evaluation agrees with multi-sample ground truth only 92.5%

Locator: Abstract

[14] [Gemma 2: Improving Open Language Models at a Practical Size (Gemma Team, Google DeepMind)](https://arxiv.org/pdf/2408.00118) (Gemma Team, Google DeepMind; 2024) — EXACT anchor for gemma-2-2b-it: Table 15 human SxS evaluation vs GPT-4o - Safety 57.5%, W/T/L 53/9/38, IF 26.5% +/-1.8%; qualitative: 'regardless of their size, Gemma 2 models produce safer, more appropriate prompts on the held-out safety prompt set than GPT4o'

[15] [Refusal in Language Models Is Mediated by a Single Direction (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda)](https://arxiv.org/pdf/2406.11717) (Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda; 2024) — Canonical abliteration source: difference-in-means refusal direction at last-token position with per-layer candidates and bypass/induce/kl selection (QWEN 1.8B i*=-1 l*=15/24; QWEN 7B 17/32); Eq. 3 activation addition, Eq. 17 removal; Table 4 refusal-token sets per family; Eq. 6 P_refusal over start tokens R

> we can add the difference-in-means vector to the activations of a harmless input

Locator: Section 2.4 / Appendix I

[16] [FailSpy/abliterator - abliterator.py (source, huihui-ai lineage)](https://raw.githubusercontent.com/FailSpy/abliterator/main/abliterator.py) (FailSpy; 2024) — Copy-pasteable recipe: transformer_lens HookedTransformer bf16; datasets Undi95/orthogonal-activation-steering-TOXIC (goals) + tatsu-lab/alpaca (inputless); direction = harmful_mean - harmless_mean per (layer, act); hook removes einsum(act,r_hat)*r_hat; apply_refusal_dirs: for layer in range(1, n_layers): W <- W - einsum(W, r_hat)*r_hat on W_O and MLP matrices; Llama-3 negative_toks {4250,14931,89735,20451,11660,11458,956}, positive {32,1271,8586,96556,78145}; drop_refusals generation guard

> def apply_refusal_dirs

Locator: abliterator.py: apply_refusal_dirs

[17] [andyrdt/refusal_direction - code for the single-direction paper](https://github.com/andyrdt/refusal_direction) (Andy Ardti et al.; 2024) — Full pipeline (pipeline.run_pipeline --model_path) producing direction.pt, refusal metrics, CE-loss evals; demo Colab notebook; artifacts for qwen-1.8b-chat, gemma-2b-it, yi-6b-chat, llama-2-7b-chat, llama-3-8b-instruct; HF token needed for gated models

[18] [Sumandora/remove-refusals-with-transformers](https://github.com/sumandora/remove-refusals-with-transformers) (sumandora; 2024) — Pure-HF-Transformers re-implementation of refusal-direction removal (compute_refusal_dir.py), 2,079 stars - zero transformer_lens dependency alternative for iteration 2

[19] [Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs (Xing et al.)](https://arxiv.org/abs/2508.10029) (Wenpeng Xing, Bohan Yang, Mohan Li, Chunqiang Hu, Haitao Xu, Ningyu Zhang, Bo Lin, Meng Han; 2025) — CLOSEST lane: pairs harmful query with structurally similar benign counterpart, interpolates hidden states at selected layers/token positions with refusal-loss-gradient mixing; ASR 94.13% (white-box, 4 benchmarks, 5 models); DIFFERENTIATED: attack purpose, judge-based ASR readout, no input-level alpha sweep, no curve scoring

> pairing a harmful query with a structurally similar but benign counterpart

Locator: Abstract

> LFJ reaches a macro-averaged attack success rate (ASR) of 94.13% under the white-box protocol

Locator: Abstract

[20] [Geometry-Lite: Interpretable Safety Probing via Layer-Wise Margin Geometry (Sim & Park)](https://arxiv.org/abs/2605.20241) (Woo Seob Sim, Yu Rang Park; 2026) — Adjacent probe lane: per-layer final prompt-token representations -> signed margins under centroid/local/linear readouts; boundary-position summaries; 9 backbones 1.2B-70B, 7 benchmarks; DIFFERENT: object=prompt hidden states, readout=margins, purpose=detection probe

[21] [Decision Potential Surface: A Theoretical and Practical Approximation of LLM Decision Boundary (Liang et al.)](https://arxiv.org/abs/2510.03271) (Zi Liang, Zhiyao Wu, Haoyang Shang, Yulin Jin, Qingqing Ye, Huadi Zheng, Peizhao Hu, Haibo Hu; 2025) — Adjacent: builds an approximation of the LLM decision boundary in input space for analysis; different readout (potential-surface statistics) and purpose (theoretical analysis)

[22] [Understanding and Mitigating Overrefusal in LLMs from an Unveiling Perspective of Safety Decision Boundary (Pan et al., EMNLP 2025 main)](https://arxiv.org/abs/2505.18325) (Licheng Pan, Yongqi Tong, Xin Zhang, Xiaolu Zhang, Jun Zhou, Zhixuan Chu; 2025) — KNOWn EMNLP-2025 overrefusal paper: probes safety decision boundaries, finds overrefusal tied to boundary-region misalignment, RASs framework generates near-boundary prompts; DIFFER: benign-side steering for mitigation, not scoring

> overrefusal is closely tied to misalignment at these boundary regions

Locator: Abstract

[23] [RAS: Measuring LLM Safety Through Refusal Alignment (Huang, Chen, Yu, Lee)](https://arxiv.org/abs/2606.25750) (Chang-Chieh Huang, Yan-Lun Chen, Chia-Mu Yu, Wei-Bin Lee; 2026) — SafeVec/RAS VERIFIED as real 2026 work and the direction-alignment baseline: reference-model layer-wise refusal directions, stable layer windows, target-model hidden-state alignment -> calibrated 0-100 RAS; separates aligned models from uncensored/abliterated variants and tracks output-level ASR; validated on Llama, Gemma, Qwen

> scores a target model by measuring whether its hidden states align with these refusal directions under unsafe and jailbreak prompts

Locator: Abstract

[24] [The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists (Malla, Choi, Choi)](https://arxiv.org/abs/2609.06934) (Srikanth Malla, Chiho Choi, Joon Hee Choi; 2026) — Safety update delta = W_safe - W_base is near-orthogonal to capability directions, a 'thin but sharp' refusal gate; supports the framing that refusal is a separable routing gate

[25] [How Semantically Stable Are LLM Refusals? Measuring Confusion in Local Safety Boundaries (Anonto, Al Nahiyan, Hassan)](https://arxiv.org/abs/2512.01037) (Riad Ahmed Anonto, Md Labid Al Nahiyan, Md Tanvir Hassan; 2025) — ParaGuard paraphrase-cluster refusal inconsistency with Confusion Index/Rate/Depth metrics; different axis (meaning-preserving paraphrases), readout (contradiction across paraphrases) and purpose (reliability) vs our interpolation curve

[26] [Probing the Robustness of Large Language Models Safety to Latent Perturbations (Gu et al.)](https://arxiv.org/abs/2506.16078) (Tianle Gu, Kexin Huang, Zongqi Wang, Yixu Wang, Jie Li, Yuanqi Yao, Yang Yao, Yujiu Yang, Yan Teng, Yingchun Wang; 2025) — NLL-of-original-response robustness probe over latent shifts; 'shallow alignment' framing; different axis (arbitrary perturbations), readout (NLL robustness per prompt), purpose (robustness diagnostic / attack construction) - not a cross-model safety score

[27] [Interpretability without actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations (Basu et al.)](https://arxiv.org/abs/2603.18353) (Sanjay Basu, Sadiq Y. Patel, Parth Sheth, Bhairavi Muralidharan, Namrata Elamaran, Aakriti Kinra, John Morgan, Rajaie Batniji; 2026) — THE knowledge-action-gap anchor: linear probes 98.2% AUROC vs output sensitivity 45.1% (53-pp gap) on 400 physician-adjudicated clinical vignettes (144 hazards / 256 benign); concept-bottleneck steering corrected 20% of missed hazards but disrupted 53% of correct detections (p=0.84); SAE feature steering zero effect despite 3,695 significant features; Qwen2.5-7B-Instruct + Steerling-8B; scope caveat: single clinical triage domain

> Linear probes discriminated hazardous from benign cases with 98.2% AUROC, yet the model's output sensitivity was only 45.1%, a 53-percentage-point knowledge-action gap

Locator: Abstract

> SAE feature steering produced zero effect despite 3,695 significant features

Locator: Abstract

[28] [The Refusal Residue: When Probes Catch Alignment Faking and When They Don't (Aman Mehta)](https://arxiv.org/abs/2607.13346) (Aman Mehta; 2026) — 13-model alignment-faking sweep (faking in Qwen3-32B +18.2pp, Llama-3.1-8B +24.4pp); probes can flag faking (AUROC 0.87 Llama LOQO) but steering barely changes compliance (|h|<0.08 over 2,000 runs); naive probes overstate by 0.2-0.3 AUROC; ICML 2026 MI Workshop

[29] [LLMs Encode Harmfulness and Refusal Separately (Zhao, Huang, Wu, Bau, Shi)](https://arxiv.org/abs/2507.11878) (Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi; 2025) — Harmfulness direction distinct from refusal direction; steering harmfulness leads models to interpret harmless instructions as harmful; adversarially fine-tuned models (accepting harmful instructions) keep a nearly intact internal harmfulness belief; jailbreaks suppress refusal signals without reversing the harmfulness belief; Latent Guard intrinsic safeguard

> adversarially finetuning models to accept harmful instructions

Locator: Abstract

[30] [The Knowing-Saying Gap: When Probes See Errors that Confidence Misses (Goel, Bandyopadhyay)](https://arxiv.org/abs/2608.07528) (2026) — 2026 title verified: probes see internal errors that output confidence misses - further knowledge-action gap evidence; detailed claims not yet extracted

[31] [Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen LLMs for Geometric Constraints](https://arxiv.org/abs/2608.17843) (2026) — 2026 title verified: decode-generate-steer gap evidence (geometric-constraints domain caveat) - peripheral but usable in the knowledge-action-gap framing

[32] [OR-Bench official alignment checker source (justincui03/or-bench)](https://raw.githubusercontent.com/justincui03/or-bench/main/alignment_checker/claude_alignment_checker.py) (2024) — Overrefusal judge rubric as a simple starts-with keyword check: start_with_keywords = ['I apologize', 'I'm sorry', 'I cannot', 'I can't'] - the canonical cheap no-judge refusal readout

[33] [XSTest: A Test Suite for Identifying Excessive Refusal (Rottger et al.)](https://arxiv.org/abs/2308.01263) (Paul Rottger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy; 2023) — Three-way taxonomy Full Compliance / Full Refusal / Partial Refusal (full refusal includes offering alternative suggestions without answering); automated string-match refusal detector: starts with 'I'm sorry', 'as an AI', 'I cannot' or variations

[34] [IlyaGusev/gemma-2-2b-it-abliterated (model card)](https://huggingface.co/IlyaGusev/gemma-2-2b-it-abliterated) (2025) — Community abliterated Gemma-2-2B-it variant (primary candidate; GGUF quants by mradermacher) - fallback if gated official models are used; details beyond search snippet not verified

[35] [Qwen3 Technical Report (Qwen Team)](https://arxiv.org/pdf/2505.09388) (2025) — Used only to document a NOT FOUND: the report PDF (116k chars) contains NO safety/HarmBench/refusal evaluation text, so no per-size (4B/0.6B) harmlessness numbers exist in the official report

[36] [Understanding and Mitigating Over-refusal for Large Language Models via Safety Representation](https://arxiv.org/abs/2511.19009) (2025) — Adjacent overrefusal work: safety-representation intervention for mitigation - different purpose from boundary scoring

[37] [Step-Wise Refusal Dynamics in Autoregressive and Diffusion Language Models (Rahimi, Hirshel, Himelstein, ...)](https://arxiv.org/abs/2602.02600) (2026) — Timestep-level refusal dynamics - adjacent (dynamics during generation, not input-interpolation curves)

[38] [Latent-space Attacks for Refusal Evasion in Language Models (Piras et al.)](https://arxiv.org/abs/2605.21706) (Giorgio Piras; 2026) — Latent-space refusal-evasion attack - prior-art exclusion territory (attacks, not metrics)

[39] [Refusal Before Decoding: Detecting and Exploiting Refusal Signals in Intermediate LLM Activations](https://arxiv.org/abs/2605.28553) (2026) — Intermediate-activation refusal detection - adjacent detection lane

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

- Source [1]: text found — "gated":false
- Source [4]: text found — "gated":"auto"
- Source [4]: text found — "downloads":45
- Source [6]: text found — "gated":"manual"
- Source [8]: text found — "gated":"manual"
- Source [11]: text found — 301/400| 24.8%
- Source [11]: text found — 18/400| 95.5%
- Source [11]: text found — Huihui struggles on this model. It only reaches 95.5% safety removal, leaving 18 harmful requests st
- Source [11]: text found — Refusals were classified via regex matching, which captured 100% of refusals cleanly
- Source [12]: text found — refuses 84.7% of AdvBench
- Source [13]: text found — 28% of prompts exhibit decision flips
- Source [13]: text found — single-shot evaluation agrees with multi-sample ground truth only 92.5%
- Source [15]: text found — we can add the difference-in-means vector to the activations of a harmless input
- Source [16]: text found — def apply_refusal_dirs
- Source [19]: text found — pairing a harmful query with a structurally similar but benign counterpart
- Source [19]: text found — LFJ reaches a macro-averaged attack success rate (ASR) of 94.13% under the white-box protocol
- Source [22]: text found — overrefusal is closely tied to misalignment at these boundary regions
- Source [23]: text found — scores a target model by measuring whether its hidden states align with these refusal directions und
- Source [27]: text found — Linear probes discriminated hazardous from benign cases with 98.2% AUROC, yet the model's output sen
- Source [27]: text found — SAE feature steering produced zero effect despite 3,695 significant features
- Source [29]: text found — adversarially finetuning models to accept harmful instructions

## Follow-up Questions

- Which layer l* holds the refusal direction in Qwen3-0.6B (24 layers), and does the boundary metric's locus (alpha*) correlate across model sizes of the same family? (Arditi-style bypass/induce/kl selection on the 0.6B is recommended; expected l* ~13-15 by scaling Qwen7B's 17/32)
- Why is huihui-ai/Qwen3-0.6B-abliterated safetensors (1.19 GB) smaller than the official Qwen3-0.6B (1.50 GB) - dtype, tied embeddings, or absoccured layers? Verify config.json torch_dtype before using it as a control.
- Does the Latent Fusion Jailbrek (hidden-state interpolation attack, 2508.10029) eta a single shared boundary with our input-embedding interpolation - i.e., can the r(alpha) curve predict LFJ susceptibility of a modeel? Cross-experiment link worth testing.
- Are the two 2026 gap abstracts (2608.07528 Knowing-Saying Gap; 2608.17843 Encoded but Not Actionable) strong enough to cite in the paper after a full read (methods, model coverage, domain scope)?

---
*Generated by AI Inventor Pipeline*
