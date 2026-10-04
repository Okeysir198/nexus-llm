# nexus-llm

On-box LLM serving for the nexus gateway — the LLM sibling of
[nexus-stt](https://github.com/Okeysir198/nexus-stt) /
[nexus-tts](https://github.com/Okeysir198/nexus-tts) /
[nexus-vector](https://github.com/Okeysir198/nexus-vector).

One swap-locking FastAPI bridge over N vLLM engines. Exactly **one engine is
AWAKE at a time**; everything else is a stopped container. All public traffic
arrives via [nexus.leanwise.ai](https://nexus.leanwise.ai) — callers send
`hosted_llm:<route key>` with a D1 virtual key; the Worker strips the prefix
and forwards the bare route key here.

```
nexus.leanwise.ai  ──AIG──▶  nexus-llm.leanwise.ai (CF tunnel) ──▶ :18083 bridge ──▶ engine containers
```

- **Port:** 18083 (host) → 8080 (bridge container)
- **Current engine:** `Qwen3.8-27B-NVFP4` (RedHatAI NVFP4/FP8 quant, ~24.7 GB)
- **Hardware contract:** DGX Spark GB10, sm_121, 128 GB unified memory

## Quick start

```bash
cp .env.example .env          # fill HF_TOKEN
docker compose up -d          # from THIS folder, always
# first request cold-wakes the engine (~2–4 min: weights + CUDA graphs)
curl -s localhost:18083/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

Public route (needs a nexus D1 virtual key):

```bash
curl -s https://nexus.leanwise.ai/v1/chat/completions \
  -H "authorization: Bearer $NEXUS_KEY" -H 'content-type: application/json' \
  -d '{"model":"hosted_llm:Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

## Layout

| Path | Role |
|---|---|
| `docker-compose.yml` | parent: bridge service + `include:` of each engine |
| `llm18083-bridge/` | FastAPI swap-lock gateway (auth-less internal, wakes/stops engines via docker.sock) |
| `llm-engines/LLM_<name>/` | one vLLM engine per folder, internal `:8000` only |
| `tests/` | benchmark + sleep/wake harness (own uv project) |

Read `CLAUDE.md` before touching anything — it carries the GB10 invariants
(no TIER1 sleep, KV pin, one-route-per-container, thermal rules) and the
3-step onboarding contract for the next model.
