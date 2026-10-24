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
| Single-stream decode | 34–36 tok/s (3 runs, 400 tok, `usage.completion_tokens`), TTFT 41–76 ms |
| Smoke tests | chat OK, tool call OK (`get_weather` parsed), thinking on/off OK |

For scale: LFM2.5-8B-A1B does 74 tok/s single-stream (1.5B active vs ~4B here),
so Bolt is the slower engine — its case is quality (instruction following,
persona, tool use per the card), not speed.

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

- Concurrency bench (`03_concurrent.py 10`) and BF16-vs-FP8 quality/speed A/B —
  sustained GPU load, run under the thermal guard (invariant 8).
- DFlash A/B with `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4`
  (`--speculative-config {"method":"dflash","model":…,"num_speculative_tokens":7}`);
  NVIDIA's recipe uses `--max-num-seqs 1`. Invariant 7 expects a net loss —
  measure before enabling.
- Q4_K_M GGUF (`…-E4B-4GB`, llama.cpp `server-cuda13-b11459`) is an option if
  VRAM is ever the constraint; not needed at ~15 GiB.
