# GDN Hybrid Attention Under a 16 MB Model Budget

*A Parameter Golf non-record submission accepted on 2026-04-04.*

## One-line result

This study tests whether the Gated Delta Net hybrid pattern from OLMo Hybrid helps a very small language model train at long context lengths. In this constrained setting, the answer was conditional but useful: full attention was better at 8k context, while the hybrid crossed over between 8k and 16k and was substantially better at 32k under the same 600-second single-H100 budget.

At 32k context, the hybrid reached a 2-seed mean `final_int8_zlib_roundtrip` score of **1.4709 val_bpb**, compared with **1.7059 val_bpb** for the full-attention baseline.

## Research question

Parameter Golf rewards models that do more with less: small parameter budgets, compressed submissions, and careful training choices. That makes it a useful testbed for a practical ML systems question:

> When context length grows, should a small model spend its budget on full quadratic attention everywhere, or should most layers use a cheaper recurrent mixer while a few layers preserve full attention?

The motivation came from OLMo Hybrid's use of Gated Delta Net (GDN) layers alongside full-attention layers. The obvious scaling argument says the hybrid should become cheaper at long sequence lengths, because most layers avoid `O(L^2)` attention. The less obvious question is whether that advantage still matters in a tiny model, after compression, with only 600 seconds of training.

The contribution of this submission is not just adding a new mixer layer. It measures where the architectural trade-off changes sign under a fixed wall-clock and compressed-model objective.

## Model comparison

The experiment compares two 9-layer models with similar parameter counts:

| Model | Mixer layout | Width | Parameters |
|---|---:|---:|---:|
| Full-attention baseline | 9 attention layers | 512 | 17.06M |
| GDN hybrid | 7 GDN layers + 2 full-attention layers | 448 | 17.42M |

Both models use the same broad training recipe: RoPE, RMSNorm, grouped-query attention where attention is used, Muon optimizer for matrix parameters, Adam-style optimization for scalar/control parameters, tied embeddings, and the same FineWeb validation protocol used by the competition.

The hybrid keeps full causal attention at layers 3 and 7. The remaining layers use a Gated Delta Net mixer implemented with `flash-linear-attention`'s `chunk_gated_delta_rule`, causal depthwise convolution on Q/K/V, learned per-head gates, and decay parameters. The hybrid width is reduced from 512 to 448 so the comparison stays near the same model-size regime.

## Experimental design

Each run used:

- **Hardware:** 1x H100 SXM
- **Wall-clock:** 600 seconds per run
- **Contexts tested:** 8k, 16k, 32k
- **Evaluation:** `final_int8_zlib_roundtrip` on the full FineWeb validation split
- **Primary score:** validation bits per byte after int8 quantization, zlib compression, reload, and evaluation

The key fairness choice was schedule scaling. The hybrid did not receive a longer training budget. Step times were measured, the available optimizer-step budget was estimated for each model within 600 seconds, and warmup/warmdown schedules were rescaled to preserve similar training-phase ratios across contexts.

This matters because the experiment is not only about per-step quality. It is about *quality reached within a fixed wall-clock budget*.

## Results

### Throughput

| Context | Baseline ms/step | Hybrid ms/step | Hybrid speed | Baseline steps in 600s | Hybrid steps in 600s |
|---:|---:|---:|---:|---:|---:|
| 8k | 594 | 757 | 0.79x | 1011 | 789 |
| 16k | 952 | 847 | 1.12x | 629 | 712 |
| 32k | 2202 | 1361 | 1.62x | 271 | 440 |

At 8k, FlashAttention was fast enough that the GDN layer overhead did not pay for itself. At 16k, the hybrid became faster. At 32k, it completed about **62% more optimizer steps** in the same wall-clock budget.

### Final validation bpb after compression round trip

| Context | Baseline val_bpb | Hybrid val_bpb | Delta, hybrid - baseline | Winner |
|---:|---:|---:|---:|---|
| 8k | 1.3507 | 1.3810 | +0.0303 | Baseline |
| 16k | 1.4353 | 1.3999 | -0.0354 | Hybrid |
| 32k | 1.7059 | 1.4709 | -0.2350 | Hybrid |

The important pattern is the crossover. The hybrid was not universally better. It became better once the context length was long enough for the compute savings to dominate the recurrent-kernel overhead.

At 32k, the result was stable across two seeds:

| Seed | Hybrid 32k val_bpb | Hybrid 32k val_loss |
|---:|---:|---:|
| 1341 | 1.4703 | 2.4825 |
| 42 | 1.4716 | 2.4847 |
| Mean | **1.4709** | **2.4836** |

## What the learning curves showed

At 32k context, the hybrid was ahead from the first validation checkpoint and kept improving after the baseline ran out of time.

| Step | Baseline val_bpb | Hybrid val_bpb |
|---:|---:|---:|
| 50 | 2.5255 | 2.1231 |
| 100 | 2.1579 | 1.9097 |
| 150 | 1.9480 | 1.8040 |
| 200 | 1.8220 | 1.6803 |
| 250 | 1.7190 | 1.5973 |
| 271 | 1.7035 | ~1.5927 |
| 300 | - | 1.5459 |
| 350 | - | 1.5071 |
| 400 | - | 1.4805 |
| 440 | - | 1.4694 |

This is the clearest evidence that the hybrid's advantage was not just a noisy final evaluation. It had both better early validation behavior and more optimization steps available inside the fixed budget.

## Interpretation

The result suggests a practical rule for this model family:

> In a 17M-parameter, 600-second, single-H100 training regime, full attention remains preferable at 8k context, but a GDN hybrid becomes preferable at 16k and above.

The mechanism appears to be a combination of two effects.

First, long-context full attention becomes expensive quickly. Replacing 7 of 9 full-attention layers with GDN layers changes the throughput picture once sequence length is large enough.

Second, the recurrent state may be a reasonable inductive bottleneck for very small models. A 17M-parameter model may not be able to make efficient use of all-to-all 32k attention in every layer. The hybrid forces most layers to compress history while still giving two layers global attention access.

That second point is a hypothesis, not a proven claim. The data supports it enough to justify follow-up experiments, but the current contribution is the measured crossover under a controlled budget.

## Next experiments

This submission is intentionally concrete, but it opens several clean next experiments:

1. **Ablate the attention layer placement.** The current hybrid uses attention at layers 3 and 7. Testing `{2, 6}`, `{4, 8}`, and a single-attention-layer variant would show whether the win comes from the GDN majority or from this exact placement.
2. **Sweep the GDN-to-attention ratio.** A 6:3 or 8:1 split might improve the 16k/32k trade-off depending on whether the model is bottlenecked by global mixing or throughput.
3. **Tune the 8k regime separately.** The hybrid had signs of better per-step loss at 8k but lost on wall-clock. A kernel-aware batch/token schedule might recover some of that.
4. **Run more seeds at 16k.** The 32k result was very stable across two seeds. The 16k crossover is smaller and deserves more replication.

## Why this is meaningful

This is a small but useful systems-oriented result: it identifies a specific context-length crossover where a modern hybrid sequence mixer becomes worthwhile under a compressed, fixed-budget training objective.

It also demonstrates several research-engineering habits that matter in model-development work:

- faithful implementation of a recent architecture idea in a different codebase;
- adaptation to a new constraint regime rather than assuming scale results transfer directly;
- controls around wall-clock, parameter count, compression, and validation protocol;
- reporting negative and conditional results, not only wins;
- public code, scripts, logs, and submission metadata.

The strongest part of the result is that it has a boundary. The hybrid did not simply "win"; it lost at 8k, crossed over around 16k, and became clearly better at 32k. That boundary makes the experiment more useful than a one-row benchmark, because it says something about when the architecture is worth its complexity.

## Code and submission

The full implementation and discussion are available in the Parameter Golf submission PR:

- [PR #1371: Non-record Olmo Hybrid GDN long-context study](https://github.com/openai/parameter-golf/pull/1371)
- [Fork branch: `aarjunsrinivasan:gdn_long_context`](https://github.com/aarjunsrinivasan/parameter-golf/tree/gdn_long_context)

The branch includes the hybrid model implementation, baseline comparison, experiment notes, logs, and submission metadata. The PR conversation is the best place to see the experiment in its original competition context.
