# nexus-llm tests

Own uv project, independent of the stack:

```bash
cp .env.example .env && uv sync
uv run python 01_streaming.py          # token-accurate streaming bench (usage.completion_tokens / wall time)
uv run python 00_sleep_wake.py         # TIER2 sleep/wake cycle (skip MODE=1 — TIER1 is off on GB10)
uv run python 02_agent.py              # tool-calling round trip
uv run python 03_concurrent.py 10      # concurrency: TTFT / req/s / agent tok/s
```

`TARGET` selects the provider (`vllm` default = `localhost:$VLLM_PORT`); the
cloud targets are comparison baselines and need their API keys in `.env`.

⚠️ Benchmarks are sustained GPU load — run them under the thermal guard in
`/srv/share/00_general/P20251225-general-setup/host-forensics/guarded-load/`.
Count tokens from `usage.completion_tokens`, never SSE chunks (spec-decode
engines pack multiple tokens per chunk and chunk-counting under-reports 4.5×).
