"""Section 0: Gateway sleep/wake + multi-model swap (per-upstream API).

Three cycles, each independently runnable:

  1 — tier-1 sleep/wake on the AWAKE model
  2 — tier-2 sleep/wake on the AWAKE model
  swap — multi-model swap: drive requests to non-AWAKE models and assert
         the invariant (|AWAKE|=1, |TIER1|≤max_tier1, LRU eviction)

Usage:
  uv run 00_sleep_wake.py              # all three (default)
  MODE=1 uv run 00_sleep_wake.py       # tier-1 only
  MODE=2 uv run 00_sleep_wake.py       # tier-2 only
  MODE=swap uv run 00_sleep_wake.py    # swap only

(env var rather than argv to avoid colliding with shared.py's positional
TARGET / concurrency parser.)
"""
from __future__ import annotations

import concurrent.futures
import os
import time

import requests

from shared import BASE, section, console, llm, SYSTEM, chat_line

GATEWAY = BASE.rsplit("/v1", 1)[0]


# ──────────────────────────────────────────────────────────────────────────
# /admin helpers
# ──────────────────────────────────────────────────────────────────────────

def get_state() -> dict:
    r = requests.get(f"{GATEWAY}/admin/state", timeout=5)
    r.raise_for_status()
    return r.json()


def current_awake() -> str | None:
    return get_state().get("current_awake")


def model_level(model: str) -> str:
    return get_state()["upstreams"][model]["sleep_level"]


def force_sleep(model: str, level: int) -> None:
    section(f"Force sleep → {model} tier-{level}")
    r = requests.post(
        f"{GATEWAY}/admin/sleep", params={"model": model, "level": level}, timeout=60
    )
    r.raise_for_status()
    console.print(f"  → {r.json()}")


def force_wake(model: str) -> None:
    section(f"Wake → {model}")
    r = requests.post(f"{GATEWAY}/admin/wake", params={"model": model}, timeout=300)
    r.raise_for_status()
    console.print(f"  → {r.json()}")


def wait_for_level(model: str, expected: str, timeout: int = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        lvl = model_level(model)
        if lvl == expected:
            return
        console.print(f"  [dim]{model} = {lvl}, waiting for {expected}…[/]")
        time.sleep(2)
    raise RuntimeError(f"{model} did not reach {expected} in {timeout}s")


def check_models() -> None:
    section("Health check", "GET /v1/models")
    r = requests.get(f"{BASE}/models", timeout=30)
    r.raise_for_status()
    models = [m["id"] for m in r.json().get("data", [])]
    console.print(f"  models: {', '.join(models) if models else '(none)'}")


def quick_chat() -> None:
    section("Quick streaming chat (default AWAKE)")
    prompt = "Say hello in one short sentence."
    console.print(chat_line("USER", prompt))
    text = ""
    for chunk in llm.stream(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": prompt}]
    ):
        if chunk.content:
            text += chunk.content
    console.print(chat_line("REPLY", text))


# ──────────────────────────────────────────────────────────────────────────
# Cycles
# ──────────────────────────────────────────────────────────────────────────

def run_tier(level: int) -> None:
    awake = current_awake()
    if awake is None:
        raise RuntimeError("no model is AWAKE — start the gateway with a default_awake")
    console.print(f"[dim]current_awake = {awake}[/]")

    force_sleep(awake, level)
    expected = "TIER1" if level == 1 else "TIER2"
    wait_for_level(awake, expected)

    force_wake(awake)
    wait_for_level(awake, "AWAKE")

    check_models()
    quick_chat()


# ── swap helpers ──────────────────────────────────────────────────────────

def chat_against(model: str, *, timeout: int = 1000) -> dict:
    """One short non-streaming chat completion against a specific upstream.

    Returns the response JSON. This is the production wire path — any
    swap orchestration must happen transparently before the body comes back.
    """
    t0 = time.monotonic()
    r = requests.post(
        f"{BASE}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say 'ok'."}],
            "max_tokens": 8,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    r.raise_for_status()
    console.print(f"  [dim]chat {model} → {elapsed:.1f}s[/]")
    return r.json()


def assert_invariant(state: dict, where: str) -> None:
    """|AWAKE| == 1, |TIER1| ≤ max_tier1."""
    awake = [m for m, u in state["upstreams"].items() if u["sleep_level"] == "AWAKE"]
    tier1 = [m for m, u in state["upstreams"].items() if u["sleep_level"] == "TIER1"]
    max_tier1 = state["max_tier1"]
    assert len(awake) == 1, f"[{where}] expected 1 AWAKE, got {awake}"
    assert len(tier1) <= max_tier1, (
        f"[{where}] TIER1={tier1} exceeds max_tier1={max_tier1}"
    )
    console.print(
        f"  [dim]invariant OK: AWAKE={awake[0]}  TIER1={tier1}  (cap={max_tier1})[/]"
    )


def print_state_table(state: dict) -> None:
    for m, u in state["upstreams"].items():
        age = u.get("last_used_age_s")
        age_s = f"{age:>6.1f}s" if age is not None else "    -- "
        console.print(
            f"  [dim]{m:<55} {u['sleep_level']:<6} in_flight={u['in_flight']} last_used={age_s}[/]"
        )


def run_swap_cycle() -> None:
    section("Multi-model swap cycle")
    state = get_state()
    catalog = list(state["upstreams"])
    max_tier1 = state["max_tier1"]
    default_awake = state["default_awake"]
    console.print(f"[dim]catalog ({len(catalog)}): {catalog}[/]")
    console.print(f"[dim]max_tier1={max_tier1}  default_awake={default_awake}[/]")

    if len(catalog) < 2:
        console.print("[yellow]skip: need ≥2 routes to exercise swap[/]")
        return

    assert_invariant(state, "boot")
    print_state_table(state)

    # ── step 1: pick first non-AWAKE target, swap to it ───────────────────
    awake_now = state["current_awake"]
    non_awake = [m for m in catalog if m != awake_now]
    target1 = non_awake[0]

    section(f"Swap 1: {awake_now} → {target1}")
    chat_against(target1)
    state = get_state()
    assert_invariant(state, "after swap 1")
    assert state["current_awake"] == target1, (
        f"expected AWAKE={target1}, got {state['current_awake']}"
    )
    # previous AWAKE should be in TIER1 (or TIER2 if max_tier1=0 — but we
    # rejected that config in the gateway).
    prev_lvl = state["upstreams"][awake_now]["sleep_level"]
    assert prev_lvl in ("TIER1", "TIER2"), f"previous AWAKE level={prev_lvl}"
    console.print(f"  [green]✓ AWAKE flipped to {target1}, previous → {prev_lvl}[/]")
    print_state_table(state)

    # ── step 2: LRU eviction (only meaningful with max_tier1 ≥ 1) ────────
    # To force LRU eviction, we need to push tier1 past max_tier1. That
    # means visiting (max_tier1 + 1) distinct non-default models, with the
    # default originally AWAKE. The very first model to sleep should be
    # the LRU pick when we make the (max_tier1+2)th swap.
    # max_tier1=0 means tier1 is always empty (every eviction → tier2),
    # so the LRU exercise doesn't apply.
    needed_for_lru = max_tier1 + 2
    if max_tier1 == 0:
        console.print(
            "  [yellow]skip LRU eviction: max_tier1=0 (tier-1 disabled by config)[/]"
        )
    elif len(catalog) >= needed_for_lru:
        section(f"LRU eviction (need {needed_for_lru} routes, have {len(catalog)})")
        # Visit the first `max_tier1 + 1` non-default models in order.
        # Each becomes AWAKE in turn; the previous one drops to TIER1.
        # After max_tier1 fills up, the next swap should demote the LRU
        # (the very first one we visited, which has the oldest last_used).
        ordered_targets = [m for m in catalog if m != default_awake][: max_tier1 + 1]
        console.print(f"  [dim]visiting in order: {ordered_targets}[/]")
        for t in ordered_targets:
            chat_against(t)
            time.sleep(0.5)  # ensure last_used timestamps are monotonically distinct
        # The earliest visited (which is now in TIER1, oldest last_used)
        # is the LRU. The NEXT swap should demote it to TIER2.
        first_visited = ordered_targets[0]
        # Pick a target we haven't visited yet — that's any catalog member
        # not in ordered_targets and not currently AWAKE.
        awake_now = get_state()["current_awake"]
        unvisited = [
            m for m in catalog
            if m not in ordered_targets and m != awake_now
        ]
        if not unvisited:
            console.print(
                "  [yellow]skip LRU eviction: not enough unvisited models[/]"
            )
        else:
            evict_trigger = unvisited[0]
            console.print(
                f"  [dim]triggering eviction by waking {evict_trigger} "
                f"(should drop {first_visited} → TIER2)[/]"
            )
            chat_against(evict_trigger)
            state = get_state()
            assert_invariant(state, "after LRU trigger")
            evicted_lvl = state["upstreams"][first_visited]["sleep_level"]
            assert evicted_lvl == "TIER2", (
                f"expected LRU {first_visited} → TIER2, got {evicted_lvl}"
            )
            console.print(
                f"  [green]✓ LRU {first_visited} demoted to TIER2[/]"
            )
            print_state_table(state)
    else:
        console.print(
            f"  [yellow]skip LRU eviction: need {needed_for_lru} routes, have {len(catalog)}[/]"
        )

    # ── step 3: concurrent-swap guard ─────────────────────────────────────
    # Two simultaneous requests for the SAME currently-non-AWAKE model.
    # Both must return 200; the gateway must serialize the single swap.
    awake_now = get_state()["current_awake"]
    concurrent_target = next((m for m in catalog if m != awake_now), None)
    if concurrent_target is None:
        console.print("[yellow]skip concurrent guard: no non-AWAKE target[/]")
    else:
        section(f"Concurrent swap guard ({concurrent_target} × 2)")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(chat_against, concurrent_target) for _ in range(2)]
            results = [f.result() for f in futs]
        assert len(results) == 2, "both concurrent requests should succeed"
        state = get_state()
        assert_invariant(state, "after concurrent swap")
        assert state["current_awake"] == concurrent_target
        console.print(
            f"  [green]✓ both concurrent requests succeeded, single AWAKE={concurrent_target}[/]"
        )

    # ── step 4: restore default_awake ─────────────────────────────────────
    section(f"Restore default_awake → {default_awake}")
    chat_against(default_awake)
    state = get_state()
    assert_invariant(state, "post-restore")
    assert state["current_awake"] == default_awake
    console.print(f"  [green]✓ restored AWAKE={default_awake}[/]")
    print_state_table(state)


if __name__ == "__main__":
    mode = os.environ.get("MODE", "all").lower()

    if mode in ("1", "all"):
        run_tier(1)

    if mode in ("2", "all"):
        run_tier(2)

    if mode in ("swap", "all"):
        run_swap_cycle()

    console.print("[bold green]All sleep/wake cycles passed.[/]")
