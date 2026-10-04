"""OpenAI-compatible front-door for self-hosted vLLM stacks.

Design rules (single page):

  1. At most 1 upstream is AWAKE (VRAM resident, serving).
  2. At most `max_tier1` upstreams are in TIER1 (vLLM `/sleep level=1`,
     weights in host RAM, ~3-4 s wake).
  3. Everything else is TIER2 (container stopped, ~20-30 s wake).
  4. Swaps are request-driven: the next request to a non-AWAKE model
     triggers a swap. An idle reaper additionally demotes the AWAKE model
     after `idle_sleep_s` of no traffic.
  5. The gateway forwards the body unchanged. The one exception is the
     Qwen chat-template shim (system-only / double-system messages), gated
     on the model id.

Onboarding a new model is two changes: drop in an `LLM_<name>/` compose
folder, add one entry to `routes:` in `config.yaml`, and restart the
gateway. Set `supports_sleep_mode: true` on the route only if the engine
runs with `--enable-sleep-mode`.

All tunables live in `config.yaml` (path from GATEWAY_CONFIG_PATH env,
default `/app/config.yaml`). `${VAR:-default}` interpolation against the
process env is applied at load time so compose env_file values flow in.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from loguru import logger


CONFIG_PATH = os.environ.get("GATEWAY_CONFIG_PATH", "/app/config.yaml")

# Chat templates assert against double-system / non-leading-system message
# arrays that cloud-OpenAI clients (and LangChain, and the Claude Code →
# Anthropic→OpenAI proxy) commonly emit. `_normalize_chat_messages` merges all
# system parts into one leading system message; this is universally safe and runs
# for EVERY chat request. Appending an empty user turn (for templates that 422 on
# system-only input) is NOT universally safe — it changes generation for models
# that accept system-only — so it is gated on this set. List a model here only if
# its template rejects system-only arrays.
TEMPLATE_REQUIRES_USER_TURN: set[str] = {"Qwen3.6-35B-A3B-NVFP4", "diffusiongemma-26b-a4b-it"}

# After `docker stop` of the previous AWAKE container, the NVIDIA driver
# needs a moment to fully reclaim VRAM (CUDA context teardown is async).
# Starting the new vLLM container during that window causes the new
# process to see leftover VRAM and OOM mid-load. 10 s is empirically
# enough; harmless on unified-memory hosts (DGX Spark) where there is no
# discrete VRAM ceiling.
_VRAM_RECLAIM_GRACE_S: float = 10.0

# How often the background resyncer re-derives every upstream's state from
# Docker, so the gateway notices containers started/stopped out-of-band
# (e.g. by the gpu-manager, or a manual `docker stop`).
_RESYNC_INTERVAL_S: float = 30.0


class SleepLevel(IntEnum):
    AWAKE = 0       # engine ready, /v1/models populated
    TIER1 = 1       # vLLM /sleep level=1: VRAM freed, weights pinned in host RAM
    TIER2 = 2       # docker stopped: VRAM + RAM freed


@dataclass(frozen=True)
class Route:
    model: str
    upstream: str           # e.g. http://Qwen3.6-35B-A3B-NVFP4:8000
    container: str          # docker container name (required for sleep/wake)
    supports_sleep_mode: bool  # whether the engine was started with --enable-sleep-mode
    # Co-resident engine (small GGUF llama.cpp servers): NOT part of the swap-lock.
    # Never evicted and never evicts another upstream — it just runs alongside the one
    # swap-managed model. Invisible to current_awake()/the |AWAKE|=1 invariant.
    always_on: bool = False


@dataclass
class Config:
    default_awake: str          # model id of the upstream that starts AWAKE
    max_tier1: int              # max simultaneously in TIER1 (host RAM)
    wake_timeout_s: int
    # Idle-demote the AWAKE upstream after this many seconds of zero
    # requests + zero in-flight. 0 disables the reaper.
    idle_sleep_s: int
    routes: dict[str, Route]    # model id → Route


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value):
    """Recursively expand `${VAR}` / `${VAR:-default}` against os.environ."""
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw = _interpolate(raw)

    routes: dict[str, Route] = {}
    for entry in raw.get("routes") or []:
        model = entry.get("model")
        upstream = (entry.get("upstream") or "").rstrip("/")
        container = entry.get("container")
        if not (model and upstream and container):
            raise ValueError(
                f"route entry missing model/upstream/container: {entry!r}"
            )
        if model in routes:
            raise ValueError(f"duplicate model id in routes: {model!r}")
        supports = entry.get("supports_sleep_mode", True)
        if isinstance(supports, str):
            supports = supports.strip().lower() in {"1", "true", "yes", "on"}
        always_on = entry.get("always_on", False)
        if isinstance(always_on, str):
            always_on = always_on.strip().lower() in {"1", "true", "yes", "on"}
        routes[model] = Route(
            model=model,
            upstream=upstream,
            container=container,
            supports_sleep_mode=bool(supports),
            always_on=bool(always_on),
        )

    default_awake = raw.get("default_awake")
    if default_awake not in routes:
        raise ValueError(
            f"default_awake={default_awake!r} not in routes "
            f"({sorted(routes)})"
        )

    return Config(
        default_awake=default_awake,
        max_tier1=int(raw.get("max_tier1", 1)),
        wake_timeout_s=int((raw.get("wake") or {}).get("timeout_s", 900)),
        idle_sleep_s=int(raw.get("idle_sleep_s", 0)),
        routes=routes,
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-upstream state
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class UpstreamState:
    route: Route
    sleep_level: SleepLevel = SleepLevel.TIER2
    in_flight: int = 0
    last_used: float = 0.0       # monotonic timestamp of last request finish
    client: httpx.AsyncClient = field(default=None)  # type: ignore[assignment]


class World:
    """Global state. One instance attached to app.state.world at lifespan."""

    def __init__(self, cfg: Config, docker: httpx.AsyncClient) -> None:
        self.cfg = cfg
        self.docker = docker
        self.switch_lock = asyncio.Lock()
        self.states: dict[str, UpstreamState] = {}
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=128)
        for model, route in cfg.routes.items():
            self.states[model] = UpstreamState(
                route=route,
                client=httpx.AsyncClient(
                    base_url=route.upstream,
                    timeout=httpx.Timeout(None, connect=10.0),
                    limits=limits,
                ),
            )

    def current_awake(self) -> str | None:
        """The single SWAP-MANAGED upstream that is AWAKE. `always_on` engines are
        co-resident (outside the |AWAKE|=1 invariant), so they are skipped here — which
        is what keeps them from ever being picked as the swap 'active' and evicted."""
        for model, st in self.states.items():
            if st.route.always_on:
                continue
            if st.sleep_level == SleepLevel.AWAKE:
                return model
        return None

    def tier1_members(self, exclude: str | None = None) -> list[str]:
        return [
            m for m, st in self.states.items()
            if st.sleep_level == SleepLevel.TIER1 and m != exclude
        ]

    async def aclose(self) -> None:
        for st in self.states.values():
            await st.client.aclose()
        await self.docker.aclose()


# ──────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────

DROP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _filter_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() not in DROP_HEADERS}


def _parse_json_body(body: bytes) -> dict | None:
    if not body:
        return None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_chat_messages(body: bytes, payload: dict, *, inject_user_turn: bool) -> bytes:
    """Rewrite the messages array so stock chat templates accept cloud-OpenAI
    patterns they assert against.

      1. Multiple / non-leading system messages → merge into one leading system
         message. Universally safe; always applied.
      2. No user message → append an empty user turn. Only when
         `inject_user_turn` (model-gated): it changes generation for models that
         accept system-only input, so it is opt-in per TEMPLATE_REQUIRES_USER_TURN.

    Best-effort: invalid shape → pass through unchanged.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return body

    original_roles = [m.get("role") for m in messages if isinstance(m, dict)]
    changed = False

    system_parts: list[str] = []
    rest: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            rest.append(m)
            continue
        if m.get("role") == "system":
            content = m.get("content")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, str):
                        system_parts.append(part)
                    elif isinstance(part, dict) and isinstance(part.get("text"), str):
                        system_parts.append(part["text"])
        else:
            rest.append(m)

    if len(system_parts) > 1 or (system_parts and original_roles[0] != "system"):
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}, *rest]
        changed = True

    if inject_user_turn and not any(
        isinstance(m, dict) and m.get("role") == "user" for m in messages
    ):
        messages.append({"role": "user", "content": ""})
        changed = True

    if not changed:
        return body

    payload["messages"] = messages
    new_roles = [m.get("role") for m in messages if isinstance(m, dict)]
    logger.warning(f"[chat-normalize] normalized roles {original_roles} -> {new_roles}")
    return json.dumps(payload).encode("utf-8")


# ──────────────────────────────────────────────────────────────────────────
# Container / engine helpers (per-upstream)
# ──────────────────────────────────────────────────────────────────────────

async def _container_running(docker: httpx.AsyncClient, name: str) -> bool:
    try:
        r = await docker.get(f"/v1.44/containers/{name}/json", timeout=5.0)
        if r.status_code != 200:
            return False
        return bool(r.json().get("State", {}).get("Running"))
    except httpx.HTTPError:
        return False


async def _container_restart_count(docker: httpx.AsyncClient, name: str) -> int | None:
    """`RestartCount` from docker inspect — bumps each time the engine crashes
    and `restart: unless-stopped` revives it. Used to fail a wake fast instead
    of polling a crash-looping container for the full wake timeout."""
    try:
        r = await docker.get(f"/v1.44/containers/{name}/json", timeout=5.0)
        if r.status_code != 200:
            return None
        return int(r.json().get("RestartCount", 0))
    except (httpx.HTTPError, ValueError, TypeError):
        return None


async def _container_action(
    docker: httpx.AsyncClient, name: str, action: str,
) -> None:
    params = {"t": 10} if action == "stop" else {}
    r = await docker.post(
        f"/v1.44/containers/{name}/{action}", params=params, timeout=120.0,
    )
    if r.status_code in (204, 304):
        return
    r.raise_for_status()


async def _is_sleeping(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get("/is_sleeping", timeout=5.0)
        r.raise_for_status()
        return bool(r.json().get("is_sleeping", False))
    except httpx.HTTPError:
        return False


async def _engine_ready(client: httpx.AsyncClient) -> bool:
    try:
        r = await client.get("/v1/models", timeout=3.0)
        return r.status_code == 200 and bool(r.json().get("data"))
    except (httpx.HTTPError, ValueError):
        return False


async def _wait_ready(
    client: httpx.AsyncClient,
    timeout_s: int,
    *,
    docker: httpx.AsyncClient | None = None,
    container: str | None = None,
    start_restarts: int | None = None,
) -> None:
    """Poll until the engine serves /v1/models. Fail FAST if the container
    crashes mid-wake (restart-count bumps, or it exits) instead of burning the
    full `timeout_s` — a crashed/looping engine would otherwise hold the swap
    lock for the whole timeout and wedge the gateway for every other request."""
    deadline = time.monotonic() + timeout_s
    grace_until = time.monotonic() + 20.0  # let `docker start` settle first
    while time.monotonic() < deadline:
        if await _engine_ready(client):
            return
        if docker and container and time.monotonic() > grace_until:
            rc = await _container_restart_count(docker, container)
            if rc is not None and start_restarts is not None and rc > start_restarts:
                raise RuntimeError(
                    f"{container} crashed during wake "
                    f"(restarted {rc - start_restarts}x) — check its logs"
                )
            if not await _container_running(docker, container):
                raise RuntimeError(f"{container} exited during wake — check its logs")
        await asyncio.sleep(2.0)
    raise TimeoutError(f"upstream unhealthy after {timeout_s}s")


async def _resync_one(world: World, model: str) -> None:
    """Re-derive sleep_level from observed reality for one upstream."""
    st = world.states[model]
    if not await _container_running(world.docker, st.route.container):
        st.sleep_level = SleepLevel.TIER2
        return
    if st.route.supports_sleep_mode and await _is_sleeping(st.client):
        st.sleep_level = SleepLevel.TIER1
        return
    if await _engine_ready(st.client):
        if st.sleep_level != SleepLevel.AWAKE:
            st.last_used = time.monotonic()
        st.sleep_level = SleepLevel.AWAKE
        return
    st.sleep_level = SleepLevel.TIER1


# ──────────────────────────────────────────────────────────────────────────
# State transitions: sleep / wake / demote
# ──────────────────────────────────────────────────────────────────────────

async def _to_tier1(world: World, model: str) -> None:
    """AWAKE → TIER1. If the upstream lacks --enable-sleep-mode, fall
    through to TIER2 (the only level it supports)."""
    st = world.states[model]
    if st.sleep_level == SleepLevel.TIER1:
        return
    if not st.route.supports_sleep_mode:
        await _to_tier2(world, model)
        return
    logger.info(f"[{model}] AWAKE → TIER1 (vLLM /sleep level=1)")
    r = await st.client.post("/sleep", params={"level": 1}, timeout=30.0)
    r.raise_for_status()
    st.sleep_level = SleepLevel.TIER1


async def _to_tier2(world: World, model: str) -> None:
    """Any → TIER2 (docker stop). Releases VRAM + host RAM."""
    st = world.states[model]
    if st.sleep_level == SleepLevel.TIER2:
        return
    logger.info(f"[{model}] {st.sleep_level.name} → TIER2 (docker stop {st.route.container})")
    await _container_action(world.docker, st.route.container, "stop")
    st.sleep_level = SleepLevel.TIER2


async def _to_awake(world: World, model: str, timeout_s: int) -> None:
    """TIER1 / TIER2 → AWAKE."""
    st = world.states[model]
    if st.sleep_level == SleepLevel.AWAKE:
        return
    t0 = time.monotonic()
    start_restarts = await _container_restart_count(world.docker, st.route.container)
    if st.sleep_level == SleepLevel.TIER2:
        logger.info(f"[{model}] TIER2 → AWAKE (docker start {st.route.container})")
        await _container_action(world.docker, st.route.container, "start")
    else:
        logger.info(f"[{model}] TIER1 → AWAKE (POST /wake_up)")
        try:
            r = await st.client.post("/wake_up", timeout=30.0)
            r.raise_for_status()
        except (httpx.ConnectError, httpx.RemoteProtocolError):
            logger.info(f"[{model}] /wake_up unreachable — engine booting, polling")
    await _wait_ready(
        st.client, timeout_s,
        docker=world.docker, container=st.route.container,
        start_restarts=start_restarts,
    )
    st.sleep_level = SleepLevel.AWAKE
    st.last_used = time.monotonic()
    logger.info(f"[{model}] AWAKE in {time.monotonic() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────────
# Swap orchestration
# ──────────────────────────────────────────────────────────────────────────

async def ensure_awake(world: World, target: str) -> None:
    """Make `target` the AWAKE upstream, enforcing |AWAKE|=1 and |TIER1|≤max_tier1."""
    st = world.states.get(target)
    if st is None:
        raise KeyError(f"unknown model: {target}")

    if st.route.always_on:
        # Co-resident engine: never swap-managed. Just make sure it is running (a
        # gpu-manager stop or crash can leave it TIER2); NEVER evict another upstream.
        if st.sleep_level != SleepLevel.AWAKE:
            async with world.switch_lock:
                if st.sleep_level != SleepLevel.AWAKE:
                    await _to_awake(world, target, world.cfg.wake_timeout_s)
        return

    if st.sleep_level == SleepLevel.AWAKE:
        return

    async with world.switch_lock:
        if st.sleep_level == SleepLevel.AWAKE:
            return

        active = world.current_awake()
        active_st = world.states[active] if active else None

        if active_st and active_st.in_flight > 0:
            logger.info(
                f"swap blocked: {active} has {active_st.in_flight} in-flight; draining"
            )
            drain_deadline = time.monotonic() + 30.0
            while active_st.in_flight > 0 and time.monotonic() < drain_deadline:
                world.switch_lock.release()
                try:
                    await asyncio.sleep(0.2)
                finally:
                    await world.switch_lock.acquire()
                if st.sleep_level == SleepLevel.AWAKE:
                    return
                active = world.current_awake()
                active_st = world.states[active] if active else None
                if active_st is None or active_st.in_flight == 0:
                    break
            if active_st and active_st.in_flight > 0:
                logger.warning(
                    f"swap proceeding while {active} has "
                    f"{active_st.in_flight} in-flight (drain timeout)"
                )

        tier1_candidates = world.tier1_members(exclude=target)
        active_can_tier1 = (
            active is not None
            and active != target
            and active_st is not None
            and active_st.route.supports_sleep_mode
        )
        if active_can_tier1:
            tier1_candidates.append(active)

        demoted_to_tier2: set[str] = set()
        while len(tier1_candidates) > world.cfg.max_tier1:
            victim = min(tier1_candidates, key=lambda m: world.states[m].last_used)
            tier1_candidates.remove(victim)
            if victim == active:
                demoted_to_tier2.add(victim)
                active_can_tier1 = False
            else:
                await _to_tier2(world, victim)
                demoted_to_tier2.add(victim)

        try:
            if active and active != target:
                if active_can_tier1:
                    await asyncio.gather(
                        _to_tier1(world, active),
                        _to_awake(world, target, world.cfg.wake_timeout_s),
                    )
                else:
                    await _to_tier2(world, active)
                    logger.info(
                        f"[{target}] waiting {_VRAM_RECLAIM_GRACE_S}s for driver "
                        f"to reclaim VRAM from {active}"
                    )
                    await asyncio.sleep(_VRAM_RECLAIM_GRACE_S)
                    await _to_awake(world, target, world.cfg.wake_timeout_s)
            else:
                await _to_awake(world, target, world.cfg.wake_timeout_s)
        except Exception:
            # Wake failed (crash, OOM, timeout). Force the target to TIER2 so a
            # crash-looping container is stopped, then re-raise — the lock is
            # released by the `async with` and the caller returns 503 instead of
            # the whole gateway hanging behind a wedged wake.
            logger.error(f"[{target}] wake failed — forcing TIER2 to free the swap lock")
            try:
                await _to_tier2(world, target)
            except Exception:
                pass
            raise


# ──────────────────────────────────────────────────────────────────────────
# Idle reaper
# ──────────────────────────────────────────────────────────────────────────

async def _idle_reaper(world: World) -> None:
    idle_s = world.cfg.idle_sleep_s
    check_interval = max(10.0, min(60.0, idle_s / 10.0))
    logger.info(
        f"idle reaper started — idle_sleep_s={idle_s}, check every {check_interval:.0f}s"
    )
    while True:
        try:
            await asyncio.sleep(check_interval)
            active = world.current_awake()
            if not active:
                continue
            st = world.states[active]
            if st.in_flight > 0:
                continue
            age = time.monotonic() - st.last_used
            if age < idle_s:
                continue
            async with world.switch_lock:
                if st.sleep_level != SleepLevel.AWAKE or st.in_flight > 0:
                    continue
                age = time.monotonic() - st.last_used
                if age < idle_s:
                    continue
                tier1_full = (
                    len(world.tier1_members(exclude=active)) >= world.cfg.max_tier1
                )
                if st.route.supports_sleep_mode and not tier1_full:
                    logger.info(
                        f"[{active}] idle {age:.0f}s ≥ {idle_s}s — demoting to TIER1"
                    )
                    await _to_tier1(world, active)
                else:
                    logger.info(
                        f"[{active}] idle {age:.0f}s ≥ {idle_s}s — demoting to TIER2"
                        + (" (sleep mode unavailable)" if not st.route.supports_sleep_mode
                           else " (tier1 full)")
                    )
                    await _to_tier2(world, active)
        except asyncio.CancelledError:
            logger.info("idle reaper stopping")
            raise
        except Exception as e:
            logger.warning(f"idle reaper iteration error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# State resyncer
# ──────────────────────────────────────────────────────────────────────────

async def _state_resyncer(world: World) -> None:
    """Re-derive every upstream's sleep_level from Docker every
    _RESYNC_INTERVAL_S — so the gateway notices containers started/stopped
    out-of-band by the gpu-manager or a manual docker call."""
    logger.info(f"state resyncer started — every {_RESYNC_INTERVAL_S:.0f}s")
    while True:
        try:
            await asyncio.sleep(_RESYNC_INTERVAL_S)
            async with world.switch_lock:
                before = {m: world.states[m].sleep_level for m in world.cfg.routes}
                await asyncio.gather(
                    *(_resync_one(world, m) for m in world.cfg.routes)
                )
                for model, prev in before.items():
                    now = world.states[model].sleep_level
                    if now != prev:
                        logger.info(
                            f"[{model}] resync: {prev.name} → {now.name} "
                            f"— changed out-of-band"
                        )
                awake = [
                    m for m in world.cfg.routes
                    if not world.states[m].route.always_on
                    and world.states[m].sleep_level == SleepLevel.AWAKE
                ]
                if len(awake) > 1:
                    logger.warning(
                        f"multiple upstreams AWAKE {awake} — GPU likely "
                        f"oversubscribed; started out-of-band?"
                    )
        except asyncio.CancelledError:
            logger.info("state resyncer stopping")
            raise
        except Exception as e:
            logger.warning(f"state resyncer iteration error: {e}")


# ──────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _load_config(CONFIG_PATH)
    docker = httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds="/var/run/docker.sock"),
        base_url="http://localhost",
        timeout=httpx.Timeout(None, connect=5.0),
    )
    world = World(cfg, docker)
    app.state.world = world

    await asyncio.gather(*(_resync_one(world, m) for m in cfg.routes))
    for model, st in world.states.items():
        logger.info(f"boot state: {model} = {st.sleep_level.name}")

    # Boot-wake is opt-out: with LLM_WAKE_ON_BOOT=false the gateway comes up
    # holding NO engine (zero VRAM); the first chat request pays the cold start.
    # default_awake still names which engine that first request targets.
    wake_on_boot = os.environ.get("LLM_WAKE_ON_BOOT", "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )
    if wake_on_boot:
        try:
            await ensure_awake(world, cfg.default_awake)
        except Exception as e:
            logger.error(f"failed to wake default_awake={cfg.default_awake}: {e}")
    else:
        logger.info(
            f"LLM_WAKE_ON_BOOT=false — not waking default_awake={cfg.default_awake} "
            f"at boot; first request will cold-start it"
        )

    logger.info(
        f"gateway up — default_awake={cfg.default_awake} wake_on_boot={wake_on_boot} "
        f"max_tier1={cfg.max_tier1} idle_sleep_s={cfg.idle_sleep_s} "
        f"routes={sorted(cfg.routes)}"
    )

    background_tasks = [asyncio.create_task(_state_resyncer(world))]
    if cfg.idle_sleep_s > 0:
        background_tasks.append(asyncio.create_task(_idle_reaper(world)))

    try:
        yield
    finally:
        for task in background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await world.aclose()


# ──────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────

API_DESCRIPTION = """
Multi-model, OpenAI-compatible gateway over self-hosted vLLM stacks.

* `/v1/chat/completions`, `/v1/completions` — routed by the `model` field.
* `/v1/models` — aggregated catalog across every configured upstream.
* Swap policy: 1 AWAKE + up to `max_tier1` in TIER1 (host RAM) + rest TIER2
  (stopped). The next request to a non-AWAKE model triggers a swap.
* The body is forwarded unchanged. The single exception is the Qwen
  chat-template shim (system-only / double-system message arrays), gated on
  the model id.
"""

OPENAPI_TAGS = [
    {"name": "openai", "description": "OpenAI-compatible inference surface."},
    {"name": "admin", "description": "Operator controls — sleep / wake / state."},
]

app = FastAPI(
    title="vLLM Gateway",
    description=API_DESCRIPTION,
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)


# ──────────────────────────────────────────────────────────────────────────
# Inference endpoints
# ──────────────────────────────────────────────────────────────────────────

def _stream_response(upstream: httpx.Response, on_close) -> StreamingResponse:
    async def body_iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            on_close()

    return StreamingResponse(
        body_iter(),
        status_code=upstream.status_code,
        headers=_filter_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def _json(payload: dict, status: int = 200, *, indent: int | None = None) -> Response:
    return Response(
        content=json.dumps(payload, indent=indent),
        status_code=status,
        media_type="application/json",
    )


def _err(status: int, msg: str) -> Response:
    return _json({"error": msg}, status)


async def _attempt(
    request: Request, world: World, target_model: str, path: str, body: bytes,
) -> tuple[UpstreamState, httpx.Response]:
    await ensure_awake(world, target_model)
    st = world.states[target_model]
    st.in_flight += 1
    try:
        upstream = await st.client.send(
            st.client.build_request(
                request.method, path,
                params=request.query_params,
                headers=_filter_headers(request.headers),
                content=body,
            ),
            stream=True,
        )
    except BaseException:
        st.in_flight -= 1
        st.last_used = time.monotonic()
        raise
    return st, upstream


async def _forward(
    request: Request,
    target_model: str,
    path: str,
    body: bytes,
) -> Response:
    world: World = request.app.state.world

    try:
        try:
            st, upstream = await _attempt(request, world, target_model, path, body)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            logger.warning(
                f"forward to {target_model} failed ({e}); resyncing state + retrying"
            )
            await _resync_one(world, target_model)
            st, upstream = await _attempt(request, world, target_model, path, body)
    except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
        logger.warning(f"forward to {target_model} unreachable after retry: {e}")
        return _err(502, f"upstream unreachable: {e}")
    except Exception as e:
        logger.error(f"ensure_awake({target_model}) failed: {e}")
        return _err(503, f"upstream {target_model} unavailable: {e}")

    def _release() -> None:
        st.in_flight -= 1
        st.last_used = time.monotonic()

    return _stream_response(upstream, _release)


async def _route_openai(request: Request, path: str, *, normalize_messages: bool) -> Response:
    world: World = request.app.state.world
    body = await request.body()
    payload = _parse_json_body(body)
    if payload is None:
        return _err(400, "invalid JSON body")
    model = payload.get("model")
    if model not in world.cfg.routes:
        return _err(404, f"unknown model: {model!r} (known: {sorted(world.cfg.routes)})")
    # System-message merge is universally safe and several clients (Claude Code
    # via the proxy) emit arrays that stock chat templates 422 on, so it runs for
    # every chat-completions request. The empty-user-turn injection is model-gated
    # (TEMPLATE_REQUIRES_USER_TURN) since it changes generation for system-only-capable models.
    if normalize_messages:
        body = _normalize_chat_messages(
            body, payload, inject_user_turn=model in TEMPLATE_REQUIRES_USER_TURN
        )
    return await _forward(request, model, path, body)


@app.post("/v1/chat/completions", tags=["openai"])
async def chat_completions(request: Request) -> Response:
    return await _route_openai(request, "/v1/chat/completions", normalize_messages=True)


@app.post("/v1/completions", tags=["openai"])
async def completions(request: Request) -> Response:
    return await _route_openai(request, "/v1/completions", normalize_messages=False)


@app.get("/v1/models", tags=["openai"])
async def list_models(request: Request) -> Response:
    world: World = request.app.state.world
    now = int(time.time())
    data = [
        {
            "id": model,
            "object": "model",
            "created": now,
            "owned_by": "selfhost",
        }
        for model in sorted(world.cfg.routes)
    ]
    return _json({"object": "list", "data": data})


# ──────────────────────────────────────────────────────────────────────────
# Admin endpoints
# ──────────────────────────────────────────────────────────────────────────

@app.get("/admin/state", tags=["admin"])
async def admin_state(request: Request) -> Response:
    world: World = request.app.state.world
    out = {
        "default_awake": world.cfg.default_awake,
        "max_tier1": world.cfg.max_tier1,
        "idle_sleep_s": world.cfg.idle_sleep_s,
        "current_awake": world.current_awake(),
        "tier1": world.tier1_members(),
        "upstreams": {
            model: {
                "sleep_level": st.sleep_level.name,
                "in_flight": st.in_flight,
                "last_used_age_s": (
                    round(time.monotonic() - st.last_used, 1)
                    if st.last_used else None
                ),
                "container": st.route.container,
                "upstream": st.route.upstream,
                "supports_sleep_mode": st.route.supports_sleep_mode,
            }
            for model, st in world.states.items()
        },
    }
    return _json(out, indent=2)


@app.post("/admin/wake", tags=["admin"])
async def admin_wake(request: Request, model: str) -> Response:
    world: World = request.app.state.world
    if model not in world.cfg.routes:
        return _err(404, f"unknown model: {model!r}")
    try:
        await ensure_awake(world, model)
    except Exception as e:
        return _err(502, f"wake failed: {e}")
    return _json({"awake": world.current_awake()})


@app.post("/admin/sleep", tags=["admin"])
async def admin_sleep(request: Request, model: str, level: int = 1) -> Response:
    world: World = request.app.state.world
    if model not in world.cfg.routes:
        return _err(404, f"unknown model: {model!r}")
    if level not in (1, 2):
        return _err(400, "level must be 1 or 2")
    st = world.states[model]
    async with world.switch_lock:
        if st.in_flight > 0:
            return _err(409, f"{model} has {st.in_flight} in-flight requests")
        try:
            if level == 1:
                if not st.route.supports_sleep_mode:
                    return _err(400, f"{model} was not started with --enable-sleep-mode")
                if st.sleep_level == SleepLevel.AWAKE:
                    await _to_tier1(world, model)
                elif st.sleep_level == SleepLevel.TIER2:
                    return _err(400, f"{model} is TIER2; wake it first if you want TIER1")
            else:
                await _to_tier2(world, model)
        except Exception as e:
            return _err(502, f"sleep failed: {e}")
    return _json({"model": model, "sleep_level": st.sleep_level.name})


@app.get("/health", tags=["admin"])
async def health() -> Response:
    return _json({"status": "ok"})
