# HySet PoC — design notes

Clean-room implementation of arXiv:2607.25718 ("Tools Are Not Islands"). This file maps
the paper to the code, records every place we deviate from or extend the paper, and
captures what we learned getting it to train.

## 1. The method in one paragraph

Each tool `t_j` has a learnable unit vector `z_j` (initialized from a frozen encoder's
embedding of its description). A candidate tool set `E` of size `m` is scored as

    F(x, E) = Σ_{a<b∈E} z_aᵀ M_m z_b   +   Σ_k softmax_k(ℓ) ℓ_k,     ℓ_k = r(x)ᵀ P z_k

The first term (`F_set`) is a query-independent co-invocation prior whose interaction
matrix depends on the set's cardinality `m` — so "weather + currency" can be a good
pair inside a 4-tool trip plan yet a bad pair on its own. The second term (`F_align`)
is attention-pooled query/tool relevance. Training contrasts the annotated set against
a pool of negative sets (InfoNCE), optionally reinforced by execution rewards. Inference
shortlists ~20 tools, enumerates every 1..5-subset (21,699 sets), and returns the argmax.

## 2. Paper → code

| Paper | Where | Notes |
|---|---|---|
| Hypergraph `H=(V,E)`, `E_M` | `data.py`, `negatives.canon` | sets are sorted index tuples |
| Eq. 2 `F = F_set + F_align` | `model.HySet.score_sets` | `return_parts=True` exposes both terms |
| Eq. 3 `F_set` with symmetric `M_m` | `HySet._f_set`, `interaction_matrices` | `M_m = (A_m + A_mᵀ)/2`; rows 0/1 are zero |
| Eq. 4–6 cross-attention pooling | `HySet.score_sets` | `alpha = softmax(ℓ)` within the set |
| Eq. 8 `L_ret` | `train.train` | mean over the batch |
| Eq. 9 `L_self` | `train.train` + `reward/feedback.py` | `Ê` = argmax over the pool; `Ê`, `ρ` detached |
| Eq. 10 regularizer | `HySet.regularizer` | `Σ_m ‖M_m‖_F²` on the symmetrized matrices |
| Eq. 11 `‖z_j‖ = 1` | `HySet.Z` | normalize-in-forward (see §3) |
| Eq. 12 singleton score | `HySet.tool_scores` | |
| Eq. 13 shortlist expansion | `SetRetriever.shortlist` | uses `M_M` (largest cardinality) |
| Eq. 14 argmax over `E_M(S)` | `SetRetriever.best_sets` | Gram-matrix trick: one `Z_S M_m Z_Sᵀ` per `m` |
| Greedy ranking `A_k` | `SetRetriever.greedy_rank` | closed-form incremental `F` |
| Alg. 1 (`N_sub`, `T_ref`, rollout cap) | `reward/feedback.ExecutionFeedback` | cache keyed by (qid, set) |
| Negative mix 50/30/20 | `negatives.NegativeMix.paper()` | 63 negatives → 31/19/13 |
| Theorem 1 | `tests/test_model.py` | Möbius order-3 terms vanish iff `M_2 = … = M_M` |
| Ablations: w/o F_set, shared M, M=I | `ModelConfig.interaction` = none / shared / identity | |
| Recall/NDCG/COMP@K | `metrics.py` | plus set exact/cover/F1/size for `Ê` |

## 3. Deviations and extensions (all deliberate)

| Topic | Paper / reference | Here | Why |
|---|---|---|---|
| Unit norm of `Z` | constraint (reference projects after each step) | `Z = normalize(z_raw)` in forward | same feasible set, no projection step, smooth |
| Init of `P` | unspecified (reference: Xavier) | `align_scale · I` (or scaled orthonormal map) | starts `ℓ` as a scaled cosine of good frozen embeddings; see §4 for the scale |
| Init of `M_m` | unspecified (reference: Xavier, symmetrized) | zeros | model starts as a pure alignment scorer; interactions are learned |
| Loss reduction | sum over batch (reference) | mean over batch | λ, η are batch-size independent |
| Negatives | size-matched / in-batch / hard | + optional `perturb` (drop/add one tool) and `model` (self-mined top sets from the current shortlist) | §4: the paper mix alone under-trains cardinality calibration on our data |
| Optimizer | AdamW 1e-4 + warmup/cosine (reference) | Adam 1e-4, optional `z_lr_mult` | simpler; early stopping does the rest |
| Early stopping | val Recall@5 per epoch | per epoch **and** every `eval_every` steps | §4.6 |
| Agent / reward | ToolLLaMA + DFSDT on StableToolBench, GPT-4 judge | LangChain agent over **mock tools** + LLM judge (Claude default, Ollama plugin); offline `sim` reward | PoC: no API mirror; provider is a plugin |
| Encoder | ToolBench BERT retriever / Qwen2.5-1.5B | `BAAI/bge-small-en-v1.5` (CPU) or hashing | CPU-only machine |

Things the paper describes that are **not** implemented yet: UltraTool, zero/few-shot
held-out tool/category splits, significance tests, ToolEval pass rate, the adaptive
negative mixture of its appendix A.8, a calibrated-cardinality variant.

## 4. Training findings (synthetic data, CPU)

Measured with `bge-small`, 4,000 training queries, K_pool=20, M=5.

1. **Validation metrics peak early, then decay, while training loss keeps falling** —
   on *training* queries too, so it is not overfitting. The pool objective can be
   minimized by the query-independent `F_set` prior alone (random and hard negatives
   are mostly "tools that never co-occur"), so `F_set` grows until it swamps the query
   signal at inference, where the competition is between near-miss sets drawn from the
   query's own shortlist. Early stopping on val Recall@5 (as the paper does) is load-bearing.
   A structural reason: the in-set softmax gives a low-relevance tool almost no weight,
   so adding an irrelevant tool barely lowers `F_align`; whether to include it is left
   to the query-independent pair terms.
2. **Model-mined negatives fix most of that.** Filling half the pool with the current
   model's top-scoring wrong sets over its own shortlist (ANCE-style) makes training see
   the same competition as inference. Validation COMP@5 went from ~0.47 (paper mix) to
   ~0.73 and exact-set match from ~0.15 to ~0.47 after one epoch. Hence the CLI default
   `--neg-mix mined` (`NegativeMix.mined()`); `--neg-mix paper` reproduces the paper mix.
3. **The regularizer λ stops the drift.** λ=1e-4 decays after ~100 steps; λ=1e-2 (the
   reference repo's value) holds Recall@5 ≈ 0.91 / COMP@5 ≈ 0.78 for the whole run;
   λ=1e-1 pushes recall/COMP higher (0.95 / 0.87) but over-includes tools (exact ≈ 0.2).
4. **The alignment softmax temperature matters.** With `P = 10·I` the in-set softmax is
   so peaked that only the top tool's `ℓ` gets gradient; scores of secondary relevant
   tools are only ever pushed *down*, and the stage-1 shortlist erodes. Scales 3–5 keep
   stage-1 recall stable; 1 makes sets too large. Default `align_scale=5`.
5. Higher learning rates (≥5e-4) collapse the stage-1 ranking within an epoch; 1e-4
   (the reference repo's value) is stable.
6. Validation once per epoch is too coarse here (the best point can be ~100 steps in);
   `eval_every` (CLI default 50 steps) validates mid-epoch and early-stops on it.

7. **Final picture (README table).** Cardinality-specific `M_m` is what makes set
   prediction work (exact-set 48.5% vs 22.2 shared / 4.3 identity / 8.7 none). But the
   set prior does not help *ranking* on this data (the no-`F_set` ablation ranks best),
   it generalizes poorly to tool combinations unseen in training, and under the lenient
   `sim` reward a per-tool top-5 beats the predicted set. Candidate fixes: an auxiliary
   per-tool loss on `ℓ` to keep stage 1 sharp, handing the agent `Ê ∪ top-k`, and
   calibrating set size; and the real test is ToolBench with a live agent.

Final numbers are in the README's results table; reproduce with the commands there.

## 5. Execution feedback

`reward/feedback.ExecutionFeedback` mirrors Algorithm 1: a subset of `N_sub` training
queries receives feedback, re-drawn every `T_ref` steps; rewards are cached per
(query, set); a global call budget caps spend. Providers:

- `sim` — offline. Success per sub-task iff a selected tool has the needed capability
  (so functionally equivalent tools get credit, which is what `L_self` is for),
  `ρ = coverage² − 0.05·(useless tools)`, 5% seeded label noise.
- `llm` — LangChain. `mode="agent"`: the agent sees only the selected tools (mocks that
  return simulated payloads), loops ≤ `max_steps`, then the judge grades the transcript
  on a 0–1 scale. `mode="judge"`: one call deciding coverage from tool descriptions.
  Failures/refusals return `None` and are skipped, not scored 0.

## 6. ToolBench

`datasets/toolbench.py` expects ToolBench's `data/` dir: `instruction/G{1,2,3}_query.json`
(+ optional `test_query_ids/<split>_test_query_ids.json`). Tool ids follow ToolBench's
function-name convention `<api>_for_<tool>`; the library is the union of all `api_list`
entries (≈13.9k APIs). It has only been exercised on a fixture in that format — run it on
the real dump before trusting numbers. Training at full scale on CPU is slow; start with
`hyset toolbench … --max-train 20000`.
