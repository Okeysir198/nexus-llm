# nexus-llm

On-box LLM serving for the nexus gateway — the LLM sibling of
[nexus-stt](https://github.com/Okeysir198/nexus-stt) /
[nexus-tts](https://github.com/Okeysir198/nexus-tts) /
[nexus-vector](https://github.com/Okeysir198/nexus-vector).

One swap-locking FastAPI bridge over N engines (vLLM / llama.cpp). Exactly
**one engine is AWAKE at a time**; everything else is a stopped container.
All public traffic arrives via [nexus.leanwise.ai](https://nexus.leanwise.ai)
— callers send `hosted_llm:<route key>` with a D1 virtual key; the Worker
strips the prefix and forwards the bare route key here.

```
nexus.leanwise.ai  ──AIG──▶  nexus-llm.leanwise.ai (CF tunnel) ──▶ :18083 bridge ──▶ engine containers
```

- **Port:** 18083 (host) → 8080 (bridge container)
- **Fleet (2026-10-04):**

| route key | engine | VRAM | role |
|---|---|---|---|
| `LFM2.5-8B-A1B` | Liquid flagship, vLLM FP8 | 13.7 GB | **default** — best TTFT & concurrency/GB |
| `Ornith-1.5-35B-A3B-NVFP4` | vLLM NVFP4 (pinned image) | 30.3 GB | heavy standard route |
| `Qwen3.6-35B-A3B-NVFP4-B12X` | unsloth requant, vLLM b12x | 37.9 GB | 3.6 speed route |
| `Ternary-Bonsai-2-27B` | PrismML llama.cpp, PTQ1_0 | 8.6 GB | ultra-low-VRAM specialist |

- **Hardware contract:** DGX Spark GB10, sm_121, 128 GB unified memory

## Quick start

```bash
cp .env.example .env          # fill HF_TOKEN
docker compose up -d          # from THIS folder, always
# first request cold-wakes the default engine (~2–4 min: weights + CUDA graphs)
curl -s localhost:18083/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"LFM2.5-8B-A1B","messages":[{"role":"user","content":"hi"}]}'
```

Public route (needs a nexus D1 virtual key):

```bash
curl -s https://nexus.leanwise.ai/v1/chat/completions \
  -H "authorization: Bearer $NEXUS_KEY" -H 'content-type: application/json' \
  -d '{"model":"hosted_llm:LFM2.5-8B-A1B","messages":[{"role":"user","content":"hi"}]}'
```

Per-engine numbers and gotchas: `<engine folder>/BENCHMARKS.md`.

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | parent: bridge service + `include:` of each engine |
| `llm18083-bridge/` | FastAPI swap-lock gateway (auth-less internal, wakes/stops engines via docker.sock) |
| `llm-engines/LLM_<name>/` | one engine per folder, internal `:8000` only |
| `tests/` | benchmark + sleep/wake harness (own uv project) |

Read `CLAUDE.md` before touching anything — it carries the GB10 invariants
(no TIER1 sleep, KV pin, one-route-per-container, thermal rules) and the
onboarding contract for the next model.
