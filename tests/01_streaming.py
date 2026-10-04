"""Section 1: Streaming chat + Section 2: bind_tools (streamed)."""
from __future__ import annotations

import statistics
import time

from shared import TOOLS, SYSTEM, llm, section, chat_line, metrics_table, fmt, console

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # 1. STREAMING — voice latency (TTFT, ITL, decode tok/s)
    # ─────────────────────────────────────────────────────────────────────────
    section("1 · STREAMING chat", "voice latency metrics")
    USER1 = "Hi, I want to check my account balance."
    console.print(chat_line("USER", USER1))

    t0 = time.perf_counter()
    ttft: float | None = None
    tok_times: list[float] = []
    text = ""
    usage: dict = {}
    for chunk in llm.stream([{"role": "system", "content": SYSTEM},
                             {"role": "user",   "content": USER1}]):
        now = time.perf_counter()
        if chunk.content:
            if ttft is None:
                ttft = now - t0
            tok_times.append(now)
            text += chunk.content
        if chunk.usage_metadata:
            usage = dict(chunk.usage_metadata)
    total = time.perf_counter() - t0
    itls = [tok_times[i] - tok_times[i-1] for i in range(1, len(tok_times))]

    console.print(chat_line("REPLY", text))

    ttft_ok = (ttft or 0) < 0.250
    itl_ok  = bool(itls) and statistics.mean(itls) < 0.040
    rows = [
        ("TTFT",     f"{fmt(ttft or 0)}     {'[green]✓[/] target <250 ms' if ttft_ok else '[yellow]⚠[/] >250 ms'}"),
    ]
    if itls:
        rows.append((
            "ITL avg/p50/p95",
            f"{fmt(statistics.mean(itls))} / {fmt(statistics.median(itls))} / "
            f"{fmt(sorted(itls)[int(len(itls)*0.95)])}     "
            f"{'[green]✓[/] target <40 ms' if itl_ok else '[yellow]⚠[/] >40 ms'}",
        ))
    rows.append(("total stream", f"{fmt(total)}    chunks={len(tok_times)}"))
    if usage:
        completion = usage.get("output_tokens", 0)
        decode_window = max(total - (ttft or 0), 1e-6)
        rows.append(("tokens", f"prompt={usage.get('input_tokens')}  "
                               f"completion={completion}  "
                               f"total={usage.get('total_tokens')}"))
        if completion > 1:
            rows.append(("decode tok/s", f"{(completion - 1) / decode_window:.1f}"))
    console.print(metrics_table(rows, title="metrics"))

    # ─────────────────────────────────────────────────────────────────────────
    # 2. bind_tools — streamed tool-call detection
    # ─────────────────────────────────────────────────────────────────────────
    section("2 · bind_tools (streamed)", "single tool-call detection")
    USER2 = "Check balance on account 8472193."
    console.print(chat_line("USER", USER2))

    t0 = time.perf_counter()
    ttft2: float | None = None
    ai = None  # accumulated AIMessageChunk
    usage2: dict = {}
    for chunk in llm.bind_tools(TOOLS).stream(
        [{"role": "system", "content": SYSTEM},
         {"role": "user",   "content": USER2}]
    ):
        if ttft2 is None and (chunk.content or chunk.tool_call_chunks):
            ttft2 = time.perf_counter() - t0
        if chunk.usage_metadata:
            usage2 = dict(chunk.usage_metadata)
        ai = chunk if ai is None else ai + chunk
    elapsed = time.perf_counter() - t0

    if ai and ai.tool_calls:
        for c in ai.tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in c["args"].items())
            console.print(chat_line("AI", f"call → {c['name']}({args})"))
    if ai and ai.content:
        console.print(chat_line("REPLY", ai.content))

    rows = [("TTFT", fmt(ttft2 or 0)),
            ("total", fmt(elapsed))]
    if usage2:
        rows.append(("tokens",
                     f"prompt={usage2.get('input_tokens')}  "
                     f"completion={usage2.get('output_tokens')}  "
                     f"total={usage2.get('total_tokens')}"))
    verdict = ("[green]✓ vLLM extracted tool_calls natively (qwen3_coder parser)[/]"
               if ai and ai.tool_calls else
               "[red]✗ tool_calls empty — parser mismatch?[/]")
    rows.append(("verdict", verdict))
    console.print(metrics_table(rows, title="metrics"))
