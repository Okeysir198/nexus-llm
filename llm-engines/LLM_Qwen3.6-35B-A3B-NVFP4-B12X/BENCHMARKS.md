# Qwen3.6-35B-A3B-NVFP4-B12X (unsloth-Fast) — benchmarks

Restored 2026-10-04 (re-deployed from a fresh unsloth download after the RedHatAI A/B (which the
route briefly served): unsloth/Qwen3.6-35B-A3B-NVFP4-Fast is ~2x faster on
this box at equal acceptable tool accuracy.

## Head-to-head (same harness, 6k prompt, via :18083 bridge)

| engine config | single | conc-10 agg | tool acc | VRAM |
|---|---|---|---|---|
| **unsloth-Fast, b12x, no spec (THIS ROUTE)** | **69.9 tok/s** (ITL 14.1 ms) | **160 tok/s** | **9P + 1E / 10 acc** | 37.9 GB |
| RedHatAI, cutlass + DSpark spec=8 | 35.4 tok/s | 81 tok/s | 9P + 1E / 10 acc | 34.1 GB |
| RedHatAI, b12x no-spec | 42.0 tok/s | 78 tok/s | 6P + 3 partial | 43.0 GB |

Why unsloth wins: the requant is engineered for vLLM/Blackwell — FP8
attention/KV scheme that hits the fast flashinfer_b12x kernel path with
cheap KV traffic. The RedHatAI checkpoint drags BF16 KV and lands on slower
paths even on b12x (42 tok/s, 43 GB).

Notes:
- b12x and MTP are mutually exclusive on this checkpoint (unquantized MTP
  expert head — b12x rejects; MTP+cutlass measured 23 tok/s = 3x worse).
- DSpark spec-on-GB10 has now LOST on three checkpoints (Bonsai, LFM*
  unservable, Qwen3.6 RedHatAI) — treat "spec helps" as false until
  measured per-model on this box.
- Weights: unsloth/Qwen3.6-35B-A3B-NVFP4-Fast in the hub-style ./hf-cache;
  re-download with HF_HUB_CACHE=./hf-cache hf download unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
