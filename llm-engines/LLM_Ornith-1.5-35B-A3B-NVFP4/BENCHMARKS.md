# Ornith-1.5-35B-A3B-NVFP4 — benchmarks (2026-10-04)

Setup: eugr/spark-vllm-b12x:**nightly-20260815** (= v0.1.dev20003, the build
MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark was validated on) + the repo's two b12x
patch overlays, 8 GiB KV pin, util ceiling 0.40, max-model-len 131072,
max-num-seqs 24, prefix caching + chunked prefill, text-only. Benches via
`tests/01_streaming.py` / `tests/03_concurrent.py 10` through the :18083
bridge. 6,167-token prompt (system+tools). GPU clock cap was OFF during
measurement (box rebooted; reapplying `nvidia-smi -lgc 300,2200` costs ~2-9%).

## Image pin is load-bearing

- `latest` on 2026-10-04 = v0.1.dev21553 (b12x 1.5.0): the recipe's patches
  DON'T apply (changed `b12x._lib.intrinsics` API → ImportError crash-loop),
  and the stock b12x W4A16 kernel hits **CUDA illegal memory access at
  batch=10** under FULL_AND_PIECEWISE CUDA graphs. Single requests work, so
  the failure only shows under concurrency.
- nightly-20260815 + patches: clean at concurrency 10 and with MTP.
- Do not float this engine to `latest` without re-running
  `03_concurrent.py 10` and the MTP A/B.

## Results (8 GiB KV pin)

| | MTP off (default) | MTP on (`--spec-method mtp --spec-tokens 1`) |
|---|---|---|
| Engine VRAM | 30.3 GB | 33.5 GB |
| KV cache tokens | 759,837 (5.80× @131k) | 657,180 (5.01× @131k) |
| Single-stream decode | 72.7 tok/s (ITL 13.5 ms) | ~98 tok/s effective (ITL 18.6 ms, ~1.9 tok/chunk) |
| TTFT @6k prompt | 1.0–1.7 s | 2.2 s |
| Conc-10 aggregate | 138 output tok/s | 105–122 tok/s |
| Conc-10 success | 10/10 | 10/10 |

Recipe's own numbers (util 0.85, 256k ctx): 86.3 tok/s single, 440 tok/s @24 —
their config trades ~100 GB of pool for KV; ours pins 8 GiB so the box keeps
room for the other services.

**Default = MTP OFF.** On this box's traffic (multi-stream agent calls) MTP
costs 12–24% aggregate throughput and TTFT; enable only for single-stream
voice-style workloads (flip the two commented flags in docker-compose.yml).

Cold boot (weights in page cache): ~6 min to healthy (load + cudagraph
capture). Autotune/capture bursts are the thermal hot spots — keep the
guarded-load watch armed over boots and benches.
