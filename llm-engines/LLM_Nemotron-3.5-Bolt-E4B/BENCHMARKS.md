# Nemotron-3.5-Bolt-E4B — onboarding notes (2026-10-24)

Checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16` (Gemma-4-E4B-it
post-trained with GRPO; 8B total / 4B effective; text-only; 64K context;
OpenMDW-1.1). Served on stock `vllm/vllm-openai:v0.32.0` (arm64), FP8 dynamic
quant at load, 4 GiB KV pin, `gemma4` tool + reasoning parsers.

## Measured on this box

| Item | Value |
|---|---|
| Weights in VRAM (FP8) | 10.58 GiB — Gemma-4 per-layer embeddings do not quantize, so FP8 saves ~5 GB of the 15.9 GB BF16, not half |
| Total engine VRAM (gpu-manager) | ~15.5 GiB |
| KV pin | 4 GiB = 159,285 tokens (2.43x at 65,536/request) |
| Cold wake (warm weights, first boot) | ~3 min (80 s weight load) |
| Single-stream decode, plain | 34–37 tok/s, TTFT 40–55 ms |
| Smoke tests | chat OK, tool call OK (`get_weather` parsed), thinking on/off OK |

## Why plain decode was slow, and the fix (2026-10-24)

Plain decode is GPU-bound at ~27 ms/step regardless of batch: 37 tok/s per
stream at 1 *and* 8 concurrent (aggregate ~300 at 8), GPU 96% util, engine CPU
~10%. Gemma-4's mixed head dims (256 sliding / 512 full) rule out FA4, so vLLM
forces `TRITON_ATTN`; it is not weight bandwidth. **DFlash speculative decoding
fixes it** (companion drafter, 580 MB, mounted at `/drafter`):

| Config (400-tok completions, `usage.completion_tokens`) | 1 stream | 8 streams, per-stream / aggregate |
|---|---|---|
| FP8, no spec | 34–37 tok/s | 38 / ~300 tok/s |
| FP8 + DFlash, 7 draft tokens | 81–96 tok/s | 70 / 384–449 tok/s |
| FP8 + DFlash, 4 draft tokens (default) | 78–95 tok/s | 66–70 / 343–443 tok/s |

Mean acceptance length 2.4–2.7; per-position acceptance 0.63 / 0.38 / 0.23 /
0.13 / 0.07, so drafts past 4 are wasted — 4 and 7 measured equal. Tool calls
and thinking verified working under spec decode. Drafter engine VRAM +0.6 GiB;
KV pin yields 120K tokens (1.84x at 64K) vs 159K without. LFM2.5 plain is
74 tok/s, so with DFlash Bolt is now the faster single-stream engine; LFM2.5
still has the higher 8-way aggregate (219–237 at conc-10 per its notes — not
re-measured here).

## Other levers tried (2026-10-24, all with DFlash 4)

| Variant | 1 stream | 8 streams, aggregate | Verdict |
|---|---|---|---|
| FP8 per-tensor + TRITON_ATTN (default) | 78–95 tok/s | 343–455 tok/s | keep |
| `--attention-backend=FLASHINFER` | 75–78 tok/s | 330–433 tok/s | worse; also forces PIECEWISE cudagraphs with spec decode |
| BF16 weights (14.8 GiB) | 48–56 tok/s | 238–287 tok/s | worse — weight bytes do matter once spec decode removes the step floor |

The first 8-way burst after boot shows TTFT ~1.6 s (Triton `kernel_unified_attention`
JIT for new batch shapes, logged by vLLM's JIT monitor); steady-state TTFT is
~40–100 ms. `--quantization fp8` is deprecated in v0.32 — the compose uses
`fp8_per_tensor`.

## Gotchas

- **Thinking is OFF by default** in this chat template despite the card saying
  otherwise. Send `chat_template_kwargs.enable_thinking=true` (and
  `skip_special_tokens=false`, as in the card's sample) to get
  `reasoning_content`. `enable_thinking=false` is honoured, so the shared
  test-harness default in `tests/shared.py` is safe for this model.
- Card sampling is temperature 1.0 / top_p 0.95; `generation_config.json` also
  sets top_k 64, which vLLM applies as the server default.
- Weights fetch: `hf download … --exclude "*.gif" --exclude "*.png"` — one
  `--exclude` per pattern (a second bare pattern is parsed as a filename).

## Not yet done

- BF16-vs-FP8 quality/speed A/B and the full `03_concurrent.py 10` — sustained
  GPU load, run under the thermal guard (invariant 8). The 8-stream numbers above
  are short (~15 s) bursts, board peaked 72°C.
- Q4_K_M GGUF (`…-E4B-4GB`, llama.cpp `server-cuda13-b11459`) is an option if
  VRAM is ever the constraint; not needed at ~15 GiB.
