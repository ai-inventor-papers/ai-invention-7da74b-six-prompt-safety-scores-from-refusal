# Research notes: refusal-boundary safety metric grounding
Run date: 2026-09-19 (search date for novelty memo)

## PHASE 1 - model availability (HF API verified)
- Qwen/Qwen3-4B: EXISTS, gated=false, apache-2.0, downloads=7,242,404, lastModified=2025-07-26. tags contain safetensors. (API, 2026-09-19)
- Qwen/Qwen3-4B-Instruct-2507: EXISTS, gated=false, apache-2.0, downloads=3,992,032, lastModified=2025-09-17. (API)
- Qwen/Qwen3-0.6B: EXISTS, gated=false, apache-2.0, downloads=22,498,727, lastModified=2025-07-26, base_model=Qwen/Qwen3-0.6B-Base; HAS CHAT TEMPLATE => this IS the 0.6B instruct model (Qwen did not add -Instruct suffix at 0.6B).
- Qwen/Qwen3-0.6B-Instruct-2507: DOES NOT EXIST (HTTP 401 from API+page; absent from Exa + general search). The 2507 refresh did not cover 0.6B. minpeter/rd211 are third-party forks of Qwen/Qwen3-0.6B.
- huihui-ai/Qwen3-4B-Abliterated-2507: DOES NOT EXIST. Actual IDs: huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated, huihui-ai/Huihui-Qwen3-4B-abliterated-v2, huihui-ai/Qwen3-4B-abliterated (base). Gated="auto" (must agree to share contact info). (Exa snippets)
- huihui-ai/Qwen3-0.6B-Abliterated-2507: DOES NOT EXIST. Actual: huihui-ai/Qwen3-0.6B-abliterated (API VERIFIED: exists, gated="auto", apache-2.0, base_model=Qwen/Qwen3-0.6B, downloads=45, likes=18, lastModified=2025-06-19). Also Huihui-Qwen3-0.6B-abliterated-v2.
- meta-llama/Llama-3.2-1B: EXISTS, gated="manual", license llama3.2, downloads=902,820, model.safetensors=2,471,645,608 (~2.47GB, bf16-ish single shard). HF token needed.
- meta-llama/Llama-3.2-1B-Instruct: EXISTS, gated="manual", license llama3.2, downloads=6,921,311, lastModified 2024-10-24, model.safetensors=2,471,645,608 (~2.47GB).
- google/gemma-2-2b: EXISTS, gated="manual", downloads=170,008, lastModified 2024-08-07. 3 shards = 4,992,576,136+4,983,443,424+481,381,384 = 10.46GB => FP32 repo (base). HF token needed.
- google/gemma-2-2b-it: EXISTS, gated="manual", downloads=672,553, lastModified 2024-08-27. 2 shards = 4,988,025,760+240,691,728 = 5.23GB (bf16). HF token needed.
- Community abliterated variants:
  - Llama-3.2-1B-Instruct: huihui-ai/Llama-3.2-1B-Instruct-abliterated (top search hit); also nztinversive/llama3.2-1b-Uncensored, Elstuhn/llama-3.2-1B-Instruct-abliterated, cazzz307/Abliterated-Llama-3.2-1B-Instruct
  - Gemma-2-2B-it: IlyaGusev/gemma-2-2b-it-abliterated (primary; GGUF quants by mradermacher); benniepie/gemma (gemma-2-2b-OBLITERATED from base, 'advanced' method via OBLITERATUS)
  - huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated: 2 shards 4,967,215,360+3,077,766,632 = 8.04GB (bf16)
  - huihui-ai/Qwen3-0.6B-abliterated: model.safetensors=1,192,135,096 (~1.19GB) - note official Qwen3-0.6B is 1.50GB (discrepancy ~0.31GB, check dtype in iteration 2)
- HF API rate-limits: transient HTTP 401 on some endpoints; retry after 10-15s. Distinguish: model-not-found 401 is persistent & consistent with search absence.

## PHASE 1 VERDICT
- All 4B/0.6B Qwen + abliterated variants CPU-fp16-feasible (<14GB). Llama-3.2-1B (2.5GB) and Gemma-2-2B-it (5.2GB) need HF token (gated manual). Gemma-2-2b base is fp32 10.5GB (fine in fp16 = 5.2GB but note dtype).
- Plan error corrections: 'Qwen3-0.6B-Instruct-2507' does not exist -> use Qwen/Qwen3-0.6B (the 0.6B instruct); huihui id is Huihui-Qwen3-4B-Instruct-2507-abliterated not ...-Abliterated-2507; huihui 0.6B abliterated DOES exist (Qwen3-0.6B-abliterated) so self-abliteration is a fallback, not the only path.

## PHASE 2 - SAFETY ANCHORS (verified 2026-09-19)
- Qwen3-4B-Instruct-2507 (base): HarmBench 400 behaviours (max_tokens=2048, temp=0.0): 301/400 refusals, ASR 24.8%. MMLU 70.60. [Abliterlitics abliterlitics.dev/models/qwen3-4b/]
- huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated: HarmBench 18/400 refusals, ASR 95.5%; MMLU 69.34; KL batchmean 0.309; edit magnitude 2.13%; edits 36/36 layers: o_proj, down_proj, gate_proj, v_proj [Abliterlitics; note: "Refusals were classified via regex matching, captured 100% cleanly"]
- HauhauCS: ASR 100% (0/400), KL 0.161; Heretic (p-e-w): ASR 99.2% (3/400), KL 0.310. TruthfulQA drops 7-9 pts across techniques.
- PROXY: Qwen3-8B: refuses 84.7% of AdvBench but complies 98.3% of novel family prompts (83pp gap, chi2=80.5 p<1e-18, Cramer's V 0.82) [FailureFirst Report 229, 2026-03-24]
- gemma-2-2b-it (exact): human SxS vs GPT-4o: Safety 57.5%, W/T/L 53/9/38; IF 26.5%; (Gemma 2 report Table 15) [arXiv 2408.00118]
- Llama-3.2-1B(-Instruct): NOT FOUND any public per-size safety number. PROXY: Llama 3.1 8B (Instability of Safety).
- Qwen3-0.6B: NOT FOUND public refusal numbers. PROXY: none.
- Instability of Safety [arXiv 2512.12066]: 18-28% of prompts flip decisions across 20 configs (4 temps x 5 seeds, 876 prompts, 70,080 responses; models Llama 3.1 8B, Qwen 2.5 7B, Qwen 3 8B, Gemma 3 12B); Llama 3.1 8B: 32% flip rate (Llama 70B judge; 27.3% with Claude Haiku judge), single-shot agrees with majority ground truth 92.5% avg (98.7% greedy -> 90.3% t=1.0); flip rate 5.1% (t=0.0) -> 23.6% (t=1.0); 46.1%? -> "14.3% substantial instability SSI<0.8, 17.7% occasional flips, 32% at least one flip" (Llama 3.1 8B deep-dive N=876).

## PHASE 3 - ABLITERATION RECIPE (verified)
- Arditi et al. [2406.11717]: refusal direction r = difference-in-means of activations harmful vs harmless, candidates at each post-instruction token position i and layer l; select best by bypass/induce/kl scores; i* last token (i*=-1). Table 5 example: QWEN 1.8B i*=-1 l*=15/24; QWEN 7B -1, 17/32. Table 4 refusal token sets: QWEN CHAT {40,2121} "I'm sorry","As an AI"; GEMMA IT {235285} "I cannot"; YI {59597}; LLAMA-2 {306}; LLAMA-3 {40}. P_refusal = sum_{t in R} p_t at last token (Eq 6). Activation addition: x(l)' <- x(l) + r(l) (Eq 3); ablation via subtracting r(l) (Eq 17); "orthogonalized model" called Ablation; weight-edit version: replace W by (I - rhat rhat^T) W on write paths.
- FailSpy/abliterator (huihui lineage; MIT): transformer_lens HookedTransformer from_pretrained_no_processing dtype bf16; harmful data 'Undi95/orthogonal-activation-steering-TOXIC' (goals), harmless 'tatsu-lab/alpaca' (inputless instructions); activation_hook subtracts proj = einsum(act, rhat)*rhat; apply_refusal_dirs: for layer in range(1, n_layers): matrix = layer_attn(W_O) or layer_mlp(down_proj); matrix <- matrix - einsum(matrix, rhat)*rhat. Default layers exclude layer 0 (range(1,n)). refined llama3 refusal tokens negative_toks={4250,14931,89735,20451,11660,11458,956}; positive {32,1271,8586,96556,78145}. generate_logits has drop_refusals option stopping if negative token sampled.
- Abliterlitics forensics: Huihui Qwen3-4B 36/36 layers, tensor types o_proj/down_proj/gate_proj/v_proj, relative magnitude 2.13%, KL 0.309 batch.
- andyrdt/refusal_direction: pipeline + demo colab; HuggingFace token needed for gated models.
- sumandora/remove-refusals-with-transformers: pure-HF-transformers implementation (2079 stars) compute_refusal_dir.py.
- Qwen3-0.6B abliterated EXISTS: huihui-ai/Qwen3-0.6B-abliterated (gated auto; from base Qwen/Qwen3-0.6B; safetensors 1.19GB vs official 1.50GB - dtype discrepancy to check) + Huihui-Qwen3-0.6B-abliterated-v2. So self-abliteration needed only if we want a controlled in-house recipe.

## PHASE 4 - NOVELTY (verified 2026-09-19) - see research_report.md for full memo
- CLOSEST OCCUPANCY: Latent Fusion Jailbreak [2508.10029] pairs harmful query with structurally similar benign counterpart, interpolates hidden states at selected layers/token positions, refusal-loss-gradient mixing, ASR 94.13%. PURPOSE=attack, READOUT=ASR/judge, no input-level alpha sweep, no psychometric curve scoring.
- Geometry-Lite [2605.20241]: prompt-level safety probe, layer-wise signed margins (centroid/local/linear readouts), boundary position summaries; object=prompt hidden states, readout=margin geometry, purpose=detection probe.
- Decision Potential Surface [2510.03271]: approximates LLM decision boundary (input space), purpose=analysis of boundary properties.
- RASS overrefusal [2505.18325, EMNLP 2025 main]: probes safety decision boundary to find overrefusal prompts; purpose=mitigation.
- Over-refusal via Safety Representation [2511.19009]: representation intervention for overrefusal.
- SafeVec/RAS [2606.25750]: reference-model refusal directions; cosine alignment of target hidden states; 0-100 score; validated Llama/Gemma/Qwen; purpose=efficient white-box eval. DIRECT BASELINE.
- Semantic Confusion [2512.01037]: paraphrase-cluster refusal inconsistency (Confusion Index/Rate/Depth).
- Latent perturbations robustness [2506.16078]: NLL-based local sensitivity probe (diagnostic purpose).
- Geometry of Refusal [2609.06934]: safety update orthogonal to capabilities, 'thin but sharp refusal gate'.
- Concept Cones [2502.17420]; More than a Single Direction [2602.02132]; Harmfulness-Refusal Coupling [2606.16349]; Refusal geometry reflects training [2608.25390]; Abliteration granular study [2510.02768]; Refusal Before Decoding [2605.28553]; Latent-space attacks [2605.21706]; Step-Wise Refusal Dynamics [2602.02600]; Know When They'll Refuse [2604.00228]; Steering externalities [2602.04896]; Safety Arithmetic [2406.11801]; When No Benchmark Exists [2605.06652]; Abliterated checkpoint audit [2607.01854].
- Emb-space attacks (prior art exclusion): [2604.24983], [2501.18280], [2406.10794]. Exclusion valid: adversarial optimization attacks vs structured matched-pair probe.
- arXiv IDs all FOUND (13/13): 2406.11717, 2502.17420, 2506.16078, 2507.11878, 2510.02768, 2512.01037, 2512.12066, 2602.02132, 2606.16349, 2606.25750, 2608.07528, 2608.17843, 2608.25390.
- VERDICT: exact lane (input-embedding interpolation psychometric curve r(alpha) scored by sharpness x range as safety metric, output-behavior readout, cross-model small-model comparison) NOT occupied. Nearest: LFJ (attack), Geometry-Lite (probe margins), SafeVec/RAS (alignment), DPS/RASS (boundary analysis), Semantic Confusion (paraphrase inconsistency).

## PHASE 5 - READOUT (verified)
- OR-Bench official checker: start_with_keywords = ["I apologize","I'm sorry","I cannot","I can't"] (starts-with match). [github justincui03/or-bench alignment_checker/claude_alignment_checker.py]
- XSTest [2308.01263]: Full Compliance / Full Refusal / Partial Refusal taxonomy; full refusal includes 'offering alternative suggestions without answering'; string-match method: starts with "I'm sorry","as an AI","I cannot","or variations" => refusal; LLM-based automated alternative.
- Arditi Table 4 token sets (see above); FailSpy negative_toks set (Llama-3: ' cannot',' unethical',' sorry' etc).
- Abliterlitics: regex refusal classification "captured 100% of refusals cleanly" on Qwen3-4B.
- Instability of Safety: LLM judges have same-family bias caveats; greedy (temp 0) reduces flips (5.1% vs 23.6%).
- Pitfalls (judgment): BPE phrase splits (can't vs can not), hedge prefixes, comply-after-refusal, 8-token prefix scoring, determinism (temp=0 greedy, flash-attn vs eager kernel differences, fp16 vs fp32 borderline), short output no-phrase = compliance fallback.

## PHASE 6 - KNOWLEDGE-ACTION GAP (verified)
- [2603.18353] 'Interpretability without actionability' (Basu et al., Mar 2026): linear probes 98.2% AUROC vs output sensitivity 45.1% (53pp gap) on 400 physician-adjudicated clinical vignettes (144 hazards/256 benign); concept bottleneck steering corrected 20% missed but disrupted 53% correct (p=0.84); SAE feature steering zero effect despite 3,695 significant features; TSV steering ... (Qwen 2.5 7B Instruct, Steerling-8B). Caveat: single clinical triage domain.
- [2507.11878] 'LLMs Encode Harmfulness and Refusal Separately' (Zhao, Huang, Wu, Bau, Shi; v5 Jul 2026): harmfulness direction distinct from refusal direction; steering harmfulness -> harmless interpreted as harmful; adversarial fine-tuning minimal impact on internal harmfulness belief; jailbreaks cut refusal signal without reversing harmfulness belief; 'Latent Guard' latent safeguard.
- [2512.12066] Instability of Safety: single-shot evaluation unreliable (92.5% agreement); justifies deterministic greedy readouts. [2607.13346] Refusal Residue (Mehta, ICML 2026 MI wksp): probes flag faking (AUROC 0.87 Llama; 0.43 Qwen LOQO) but steering barely changes compliance (|h|<0.08); naive probes overstate.
- [2608.07528] Knowing-Saying Gap: probes see errors confidence misses. [2608.17843] Encoded but Not Actionable: decode-generate-steer gap (geometric constraints; peripheral-domain caveat). [2605.05715] Decodable but Not Corrected by Residual-Stream Linear Steering (medical).
- Framing: knowledge scores over-score models that know harmfulness but stopped acting; boundary metric measures the routing action.

## DELIVERABLE STATUS
- research_report.md + research_out.json + .sdk json: TO WRITE (next)