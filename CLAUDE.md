# nexus-llm — CLAUDE.md

On-box LLM for the nexus gateway. Own repo. SELF-CONTAINED, like
`nexus-stt` / `nexus-tts`: nothing here references the monorepo.

## Public surface — already wired, don't rebuild it

`nexus.leanwise.ai` (Worker `nexus-gateway`) routes chat by model prefix:
`hosted_llm:<route key>` (alias `hosted_vllm:`) → Cloudflare AI Gateway custom
provider `custom-nexus-llm` → `https://nexus-llm.leanwise.ai` (CF tunnel →
host port 18083) → this bridge, which receives the **bare route key**.

- Adding an on-box model needs **zero Worker changes** — `hosted_llm:<any>` is
  a wildcard; the route key just has to exist in `config.yaml` routes.
- NEVER point an AIG provider at `nexus.leanwise.ai` — that loops the Worker
  onto itself (documented loop bug in nexus-gateway `src/upstream/engine.ts`).
- The tunnel ingress targets host port **18083**. Changing
  `LLM_GATEWAY_PORT` here silently breaks the public edge.
- ⚠️ AIG forwards **no bearer** for `upstream:"engine"` providers — the
  bridge's :18083 is currently unauthenticated on the public tunnel. Planned
  fix (same as STT/TTS `ENGINE_KEY`): set `secret: "ONBOX_LLM_KEY"` on the
  `hosted_llm` provider + require it here. Not done yet.

## Layout & naming invariant

```
nexus-llm/
├── docker-compose.yml        # bridge service + `include:` per engine — RUN COMPOSE FROM HERE
├── llm18083-bridge/          # gateway.py (swap lock), config.yaml (routes), Dockerfile
├── llm-engines/LLM_<name>/   # one vLLM engine per folder; hf-cache/, vllm-cache/ (gitignored)
└── tests/                    # uv project: streaming bench, sleep/wake cycle
```

**FRESH invariant:**

```
folder  llm-engines/LLM_<X>
service <X>          container_name <X>          (= Docker DNS name)
bridge routes[].model = <X>  (route key; MUST be in --served-model-name)
edge     hosted_llm:<X>
```

- **ONE ROUTE PER CONTAINER.** Two route keys → one container breaks the swap
  state machine (endless start/stop fight). Alias above the bridge, never here.

## GB10 invariants (violating any of these OOMs or wedges the box)

1. **No TIER1 sleep.** Unified 128 GB pool: `/sleep level=1` spills weights
   into the pool it just freed and pins ~8 GB. Keep `LLM_MAX_TIER1=0`, every
   `*_SLEEP_MODE=false`, never add `--enable-sleep-mode`. TIER2 (full
   `docker stop`) is the only eviction.
2. **Pin KV absolutely, never trust `--gpu-memory-utilization`.** util × total
   profiling is non-deterministic on unified memory (crash-loops or
   over-allocates). Every engine gets a `--kv-cache-memory-bytes` pin + a
   generous util ceiling (0.40). Verify on boot log:
   `GPU KV cache size: N tokens` — if it swings wildly across boots, the pin
   isn't being applied.
3. **`HF_HUB_OFFLINE=1` once weights are cached** (`.env`). Otherwise every
   wake makes an HF metadata call; a slow one **holds the swap lock and
   wedges the whole bridge** (all requests hang until wake timeout).
4. **One awake at a time**, enforced by the bridge's swap lock (docker.sock).
   Don't `docker start` an engine by hand — that bypasses the lock and OOMs
   the pool. Use the bridge (a request) or the admin endpoint.
5. **NVFP4 weights are Blackwell-only.** sm_121. Fine here, don't port to Ada.
6. **MoE backend:** `flashinfer_b12x` + `CUTE_DSL_ARCH=sm_121a` on
   `vllm/vllm-openai:v0.25.0-aarch64` — the only healthy NVFP4-MoE path on
   sm_121. **Never marlin** (2.5× slower for W4A4 on this box). Don't force
   `--attention-backend=flash_attn` (rejects FP8 KV; vLLM picks FlashInfer).
7. **Spec-decode is a net loss on GB10 — verified on 3 checkpoints**
   (Bonsai DSpark, Ornith MTP, RedHatAI Qwen3.6-35B DSpark=8: 35 vs 69
   tok/s on the unsloth requant). Ornith's MTP head is unquantized → b12x
   rejects it (crash-loop); the LFM2.5 DSpark drafter is unservable
   (`Lfm2DSparkDraftModel` unregistered in every vLLM build here, incl.
   cu134-nightly 2026-10-04 — llama.cpp-only). The one winning combo
   (Qwen3.6-B12X + DFlash, FULL cudagraphs) is documented in the old
   DFlash recipe. Always benchmark before/after; count tokens via
   `usage.completion_tokens`, NEVER SSE chunks — spec engines pack
   multiple tokens per chunk.
8. **Sustained GPU load = thermal power-off risk** (see
   `/srv/share/00_general/CLAUDE.md`). Benchmarks/concurrency runs need the
   thermal guard (`P20251225-general-setup/host-forensics/guarded-load/`).
   Single requests are fine.
9. **Docker address pools are exhausted** — this stack pins
   `172.16.26.0/24` (stt 20/21/23, tts 22, ocr 24/25). Pin the next free /24
   if you clone this pattern.
10. **Set CPU ceilings on engines** (`cpus:` + `OMP_NUM_THREADS`) — an
    unbounded tokenizer/OMP pool starves the ~58 other containers during
    weight load / CUDA-graph capture.

## Bridge mechanics — don't regress

- `_state_resyncer` re-derives container state from Docker every 30 s — this
  is what makes sharing the host with `nexus-gpu-manager` safe. Keep it.
- Wake fail-fast: `_wait_ready` watches `RestartCount`/running state and
  raises within seconds; failed wakes are forced to TIER2. Keep it — without
  it one bad wake holds the swap lock for `wake.timeout_s` (900 s) and wedges
  everything.
- Cold wake (warm caches): Bonsai ~5 s, LFM2.5 ~2–3 min, Qwen3.6-B12X
  ~4–5 min, Ornith ~5–6 min (weight load + autotune; first-ever boot adds
  torch.compile). `start_period: 1200s` covers it — don't shorten.
- Idle reaper (`LLM_IDLE_SLEEP_S`, default 600) demotes the awake engine to
  TIER2 during quiet windows; next request pays the wake.
- Test sleep/wake:
  `curl -X POST 'localhost:18083/admin/sleep?model=LFM2.5-8B-A1B&level=2'`
  (always pass `level=2` — the handler defaults to 1).

## Onboarding contract — add the next self-hosted LLM in 3 steps

1. `cp -r llm-engines/LLM_Qwen3.6-35B-A3B-NVFP4-B12X llm-engines/LLM_<new>/`
   (or any existing engine folder; do NOT copy its hf-cache/vllm-cache)
   - rename service/container to `<new>`'s suffix; served name = route key =
     container name (carry the quantization in the name, no vendor tag)
   - download weights into its `hf-cache/` (`hf download <repo> --local-dir …`)
   - set `--kv-cache-memory-bytes` + generous `--gpu-memory-utilization`
     (mandatory, see invariant 2); copy the GB10 env block
     (`CUTE_DSL_ARCH`, `OMP_NUM_THREADS`, CPU limit); size `--max-model-len`
     to what the KV pin affords, not the checkpoint max
   - add `LLM_<PREFIX>_*` knobs to `.env` + `.env.example`
2. Add one `- llm-engines/LLM_<new>/docker-compose.yml` to the parent
   `include:` and one `routes:` entry to `llm18083-bridge/config.yaml`.
3. `docker compose up -d --no-deps <svc>` **from this repo root** — never from
   inside `LLM_*/` (the engine composes have no own `networks:` block there;
   running standalone lands them on `<folder>_default` where the bridge can't
   resolve them, and every request times out at the 900 s wake budget).
   Then wake it with a request.
4. Register it in the gpu-manager registry (nexus-gpu-manager, :18063):
   `POST /registry/services` with the compose file/project/service — and
   `DELETE /registry/services/<name>` for any engine this replaces
   (stop its container first).

Zero changes to nexus-gateway, zero bridge-code changes. Env-var contract is
the API: every knob needs a compose default + a `config.yaml` default + a
`.env.example` line. The stack must boot with an empty `.env`.

## Fleet (2026-10-04) — LFM2.5 default, Ornith standard, Bonsai specialist

`LLM_DEFAULT_MODEL=LFM2.5-8B-A1B` (Liquid flagship: FP8 weights + 4 GiB KV
pin, 13.7 GB, 74 tok/s single, 219–237 tok/s @conc-10 — best TTFT and
concurrency per GB). `Ornith-1.5-35B-A3B-NVFP4` is the standard heavy route
(30.3 GB, pinned to eugr `nightly-20260815` + 2 b12x patch overlays — NEVER
float it to `latest`, which crash-loops b12x). `Ternary-Bonsai-2-27B` is the
ultra-low-VRAM specialist (8.6 GB, PrismML llama.cpp fork built from source —
see its Dockerfile/BENCHMARKS.md). `Qwen3.6-35B-A3B-NVFP4-B12X` is the
unsloth-Fast requant (38 GB, b12x no-spec — b12x and MTP are mutually
exclusive on this checkpoint; the RedHatAI/DSpark A/B lost and was removed,
see its BENCHMARKS.md). Qwen3.8-27B and Qwen3-30B were decommissioned
(weights deleted).

Per-engine details, gotchas (image pins, the lfm2 enable_thinking trap, the
Ornith concurrency crash on `latest`), and all numbers: each engine folder's
`BENCHMARKS.md`. Test harness note: `tests/shared.py` extra_body must stay
model-conditional — `enable_thinking=false` is Qwen3-family only.

## Tests

`cd tests && cp .env.example .env && uv sync`, then:
`uv run python 01_streaming.py` (token-accurate streaming bench),
`uv run python 00_sleep_wake.py` (TIER2 cycle; skip MODE=1, TIER1 is off),
`uv run python 03_concurrent.py 10`. Benchmarks are sustained GPU load —
run them under the thermal guard (invariant 8). Target a specific engine
with `VLLM_MODEL=<route key>` (e.g. `VLLM_MODEL=LFM2.5-8B-A1B`).
`uv run python 02_agent.py` exercises the tool-calling loop (uses the same
model-conditional extra_body from shared.py).
