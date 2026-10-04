"""Section 4: Concurrent streaming chats + Section 5-6: Concurrent agent loops + comparison."""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from rich.table import Table

from shared import (
    CONC, TOOLS, SYSTEM, SYSTEM_SEQUENTIAL, SYSTEM_PARALLEL,
    llm, llm_tools,
    section, fmt, metrics_table, chat_line, console,
)

if __name__ == "__main__":
    # ─────────────────────────────────────────────────────────────────────────
    # 4. CONCURRENT streaming — measure server-side load behavior
    # ─────────────────────────────────────────────────────────────────────────
    PROMPTS = [
        "Hi, I want to check my account balance.",
        "Can you transfer me to billing?",
        "When does my subscription renew?",
        "I'd like to cancel my plan.",
        "My internet has been down all morning.",
        "Why was I charged twice last month?",
        "Can you update my mailing address?",
        "What are your business hours today?",
        "I forgot my account password.",
        "Is my package out for delivery yet?",
        "How do I upgrade to the premium tier?",
        "I need to dispute a transaction.",
        "Can you book me an appointment for tomorrow?",
        "What's the status of my refund?",
        "Tell me about my data usage this cycle.",
        "I want to add a second line to my plan.",
    ]

    def one_stream(idx: int, prompt: str) -> dict:
        """Fire one streaming chat completion; return per-request metrics."""
        t0 = time.perf_counter()
        ttft = None
        tok_times: list[float] = []
        text = ""
        usage: dict = {}
        for chunk in llm.stream(
            [{"role": "system", "content": SYSTEM},
             {"role": "user",   "content": prompt}]
        ):
            now = time.perf_counter()
            if chunk.content:
                if ttft is None:
                    ttft = now - t0
                tok_times.append(now)
                text += chunk.content
            if chunk.usage_metadata:
                usage = dict(chunk.usage_metadata)
        total = time.perf_counter() - t0
        itls = [tok_times[i] - tok_times[i - 1] for i in range(1, len(tok_times))]
        return {
            "idx": idx,
            "prompt": prompt,
            "ttft": ttft or 0,
            "total": total,
            "itl_avg": statistics.mean(itls) if itls else 0,
            "tokens_in":  usage.get("input_tokens", 0),
            "tokens_out": usage.get("output_tokens", 0),
            "reply": text,
        }

    def run_concurrent(n: int) -> tuple[list[dict], float]:
        """N parallel streaming requests via thread pool (sync streams report usage)."""
        with ThreadPoolExecutor(max_workers=n) as pool:
            t0 = time.perf_counter()
            futures = [pool.submit(one_stream, i, PROMPTS[i % len(PROMPTS)])
                       for i in range(n)]
            results = [f.result() for f in futures]
            wall = time.perf_counter() - t0
        return results, wall

    section(f"4 · CONCURRENT × {CONC}", "parallel streaming chats")
    # warmup: one tiny request so the first concurrent burst isn't cold
    console.print("[dim](warming up…)[/]")
    one_stream(-1, "ping")
    results, wall = run_concurrent(CONC)

    # Per-request table
    per_req = Table(title="per-request", show_lines=False, expand=False,
                    header_style="bold", border_style="bright_black")
    per_req.add_column("#", justify="right", style="dim")
    per_req.add_column("TTFT", justify="right")
    per_req.add_column("Total", justify="right")
    per_req.add_column("ITL avg", justify="right")
    per_req.add_column("toks in/out", justify="right")
    per_req.add_column("reply (truncated)", overflow="ellipsis")
    for r in results:
        per_req.add_row(
            str(r["idx"]),
            fmt(r["ttft"]),
            fmt(r["total"]),
            fmt(r["itl_avg"]),
            f"{r['tokens_in']}/{r['tokens_out']}",
            r["reply"][:60].replace("\n", " "),
        )
    console.print(per_req)

    # Aggregate
    ttfts  = sorted(r["ttft"]  for r in results)
    totals = sorted(r["total"] for r in results)
    toks_out_total = sum(r["tokens_out"] for r in results)
    def p50(xs): return xs[len(xs) // 2]
    def p95(xs): return xs[max(0, int(len(xs) * 0.95) - 1)]

    console.print(metrics_table([
        ("concurrency",     str(CONC)),
        ("wall clock",      fmt(wall)),
        ("TTFT avg/p50/p95/max",
            f"{fmt(statistics.mean(ttfts))} / {fmt(p50(ttfts))} / "
            f"{fmt(p95(ttfts))} / {fmt(max(ttfts))}"),
        ("total avg/p50/p95/max",
            f"{fmt(statistics.mean(totals))} / {fmt(p50(totals))} / "
            f"{fmt(p95(totals))} / {fmt(max(totals))}"),
        ("aggregate throughput",
            f"{CONC / wall:.1f} req/s   "
            f"{toks_out_total / wall:.0f} output tok/s"),
    ], title="aggregate"))

    # ─────────────────────────────────────────────────────────────────────────
    # 5. CONCURRENT AGENT LOOPS — tools + multi-turn under load
    # ─────────────────────────────────────────────────────────────────────────
    AGENT_PROMPTS: list[tuple[str, set[str]]] = [
        # ── complex multi-tool prompts (3-5 tools each) ────────────────────
        ("Account 8472193 — pull my balance, my plan, my data usage this cycle, "
         "then text me the pay-bill link.",
         {"get_account_balance", "get_plan_details", "get_data_usage",
          "send_sms_link"}),

        ("My internet is slow on account 1000001 in ZIP 60601. Check for an "
         "outage, log a service complaint, and transfer me to tech support.",
         {"check_outage_status", "log_complaint", "transfer_call"}),

        ("I want to dispute a charge on account 8472193. Show me my balance, "
         "log a billing complaint, schedule me a callback tomorrow at 3 PM "
         "about the dispute, and transfer me to billing.",
         {"get_account_balance", "log_complaint", "schedule_callback",
          "transfer_call"}),

        ("Account 1000001 — what plan am I on, how much data have I used, and "
         "text me the plan-compare link so I can see if I should upgrade.",
         {"get_plan_details", "get_data_usage", "send_sms_link"}),

        ("Account 8472193: pull my balance, check outage in ZIP 98101, log a "
         "service complaint, schedule a callback for tomorrow 10 AM about an "
         "upgrade, and transfer me to retention.",
         {"get_account_balance", "check_outage_status", "log_complaint",
          "schedule_callback", "transfer_call"}),

        ("Hi, account 1000001 — send me both the app download link and the "
         "pay-bill link, then check my balance and transfer me to sales.",
         {"send_sms_link", "get_account_balance", "transfer_call"}),

        ("I missed your call about account 8472193. Show me my balance and "
         "current plan, log a complaint about poor reception, and book me a "
         "callback tomorrow at 2 PM.",
         {"get_account_balance", "get_plan_details", "log_complaint",
          "schedule_callback"}),

        ("Account 1000001 — full account review please: balance, plan, data "
         "usage, any outages in ZIP 60601, then transfer me to billing.",
         {"get_account_balance", "get_plan_details", "get_data_usage",
          "check_outage_status", "transfer_call"}),

        ("I'm unhappy with service. Account 8472193, log a complaint, schedule "
         "a callback tomorrow at 11 AM, and text me the plan-compare link.",
         {"log_complaint", "schedule_callback", "send_sms_link"}),

        ("Account 1000001 — check outage in ZIP 98101, my data usage, log a "
         "service complaint, then text me the outage-status link.",
         {"check_outage_status", "get_data_usage", "log_complaint",
          "send_sms_link"}),

        # ── simple single-tool prompts (kept for baseline coverage) ────────────
        ("Look up balance for account 8472193 and transfer me to billing.",
         {"get_account_balance", "transfer_call"}),
        ("Check account 1000001 balance please.",
         {"get_account_balance"}),
        ("I want to cancel my plan, my account is 8472193. Transfer me to retention.",
         {"get_account_balance", "transfer_call"}),
        ("What plan am I on? Account 1000001.",
         {"get_plan_details"}),
        ("How much data have I used this month on 8472193?",
         {"get_data_usage"}),
        ("Hi, my internet is down. Account 1000001, ZIP 60601.",
         {"check_outage_status"}),
        ("Please check the balance on account 8472193 and text me a pay-bill link.",
         {"get_account_balance", "send_sms_link"}),
        ("Transfer me to billing — account 1000001.",
         {"transfer_call"}),
        ("I want to dispute a charge on account 8472193 and log it as a complaint.",
         {"get_account_balance", "log_complaint", "transfer_call"}),
        ("Schedule a callback for account 1000001 tomorrow at 2:30 PM about my plan options.",
         {"schedule_callback"}),
    ]

    agent: object  # set by run_bench

    def one_agent_run(idx: int, prompt: str, expected: set[str] | None = None) -> dict:
        """Run a full create_agent loop; collect step count, tokens, latency,
        and the names of every tool the agent actually invoked. Score against
        the optional `expected` set.

        Two latency metrics:
          - first_action_at: time to the first AI chunk of any kind
            (tool_call or content). Reflects "agent has started thinking".
          - voice_ttft: time to the first CONTENT chunk of the final AI
            message (the one with no further tool_calls). This is what the
            user actually hears in a voice agent.
        """
        t0 = time.perf_counter()
        first_action_at: float | None = None
        voice_ttft: float | None = None
        cur_step_first_content: float | None = None  # first content of the current AI step
        ai_steps = 0
        tokens_in = 0
        tokens_out = 0
        final_reply: str = ""
        actual_tools: list[str] = []

        for ev_type, payload in agent.stream(
            {"messages": [HumanMessage(prompt)]},
            stream_mode=["messages", "updates"],
        ):
            elapsed = time.perf_counter() - t0
            if ev_type == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") != "model":
                    continue
                if first_action_at is None and (
                    chunk.content or getattr(chunk, "tool_call_chunks", None)
                ):
                    first_action_at = elapsed
                # track first CONTENT chunk of the current AI step (not tool_call_chunks)
                if chunk.content and cur_step_first_content is None:
                    cur_step_first_content = elapsed
                if chunk.usage_metadata:
                    tokens_in  += chunk.usage_metadata.get("input_tokens", 0)
                    tokens_out += chunk.usage_metadata.get("output_tokens", 0)
            elif ev_type == "updates":
                for _, diff in payload.items():
                    for msg in (diff or {}).get("messages") or []:
                        if isinstance(msg, AIMessage):
                            ai_steps += 1
                            if msg.tool_calls:
                                for c in msg.tool_calls:
                                    actual_tools.append(c["name"])
                            # If THIS AI step had spoken content and no further
                            # tool calls, it IS the voice reply. Capture its
                            # first-content timestamp as the user-perceived TTFT.
                            if msg.content and not msg.tool_calls:
                                voice_ttft = cur_step_first_content
                                final_reply = msg.content
                            # next AI step starts fresh
                            cur_step_first_content = None
        total = time.perf_counter() - t0

        actual_set = set(actual_tools)
        if expected is None:
            verdict = "n/a"
            missing: set[str] = set()
            extra:   set[str] = set()
        else:
            missing = expected - actual_set
            extra   = actual_set - expected
            if not missing and not extra:
                verdict = "perfect"
            elif not missing:
                verdict = "extra"        # called all expected + extras
            elif not extra and missing < expected:
                verdict = "partial"      # got some but not all
            else:
                verdict = "wrong"        # missing required tools (and possibly extras)

        return {
            "idx": idx,
            "prompt": prompt,
            "ttft": voice_ttft or 0,                # what the user hears first
            "first_action": first_action_at or 0,   # first internal AI chunk
            "total": total,
            "steps": ai_steps,
            "tools": len(actual_tools),
            "tool_names": actual_tools,
            "expected": sorted(expected) if expected else [],
            "missing": sorted(missing),
            "extra":   sorted(extra),
            "verdict": verdict,
            "in": tokens_in,
            "out": tokens_out,
            "ok": bool(final_reply),
            "reply": final_reply,
        }

    def run_concurrent_agents(n: int) -> tuple[list[dict], float]:
        with ThreadPoolExecutor(max_workers=n) as pool:
            t0 = time.perf_counter()
            futures = []
            for i in range(n):
                prompt, expected = AGENT_PROMPTS[i % len(AGENT_PROMPTS)]
                futures.append(pool.submit(one_agent_run, i, prompt, expected))
            results = [f.result() for f in futures]
            wall = time.perf_counter() - t0
        return results, wall

    VERDICT_ICON = {
        "perfect": "[green]✓ perfect[/]",
        "extra":   "[yellow]+ extra[/]",
        "partial": "[yellow]~ partial[/]",
        "wrong":   "[red]✗ wrong[/]",
        "n/a":     "[dim]n/a[/]",
    }

    def render_agent_table(results: list[dict]) -> Table:
        t = Table(title="per-request", show_lines=False, expand=False,
                  header_style="bold", border_style="bright_black")
        t.add_column("#", justify="right", style="dim")
        t.add_column("TTFT (voice)", justify="right")
        t.add_column("1st action", justify="right", style="dim")
        t.add_column("Total", justify="right")
        t.add_column("Steps", justify="right")
        t.add_column("Tools called", overflow="fold")
        t.add_column("Expected", overflow="fold", style="dim")
        t.add_column("Verdict", overflow="fold")
        t.add_column("toks in/out", justify="right")
        t.add_column("final reply (truncated)", overflow="ellipsis")
        for r in results:
            diff_note = ""
            if r["missing"]:
                diff_note += f"\n[red]missing:[/] {', '.join(r['missing'])}"
            if r["extra"]:
                diff_note += f"\n[yellow]extra:[/] {', '.join(r['extra'])}"
            t.add_row(
                str(r["idx"]),
                fmt(r["ttft"]),
                fmt(r["first_action"]),
                fmt(r["total"]),
                str(r["steps"]),
                ", ".join(r["tool_names"]) or "[dim](none)[/]",
                ", ".join(r["expected"]) or "[dim](none)[/]",
                VERDICT_ICON.get(r["verdict"], r["verdict"]) + diff_note,
                f"{r['in']}/{r['out']}",
                (r["reply"] or "(no final reply)")[:50].replace("\n", " "),
            )
        return t

    def aggregate(results: list[dict], wall: float) -> dict:
        a_ttfts   = sorted(r["ttft"]         for r in results)
        a_actions = sorted(r["first_action"] for r in results)
        a_totals  = sorted(r["total"]        for r in results)
        n = len(results)
        return {
            "n":         n,
            "wall":      wall,
            "ok":        sum(1 for r in results if r["ok"]),
            "perfect":   sum(1 for r in results if r["verdict"] == "perfect"),
            "extra":     sum(1 for r in results if r["verdict"] == "extra"),
            "partial":   sum(1 for r in results if r["verdict"] == "partial"),
            "wrong":     sum(1 for r in results if r["verdict"] == "wrong"),
            "ttft_avg":  statistics.mean(a_ttfts),
            "ttft_p50":  p50(a_ttfts),
            "ttft_p95":  p95(a_ttfts),
            "ttft_max":  max(a_ttfts),
            "act_avg":   statistics.mean(a_actions),
            "act_p95":   p95(a_actions),
            "tot_avg":   statistics.mean(a_totals),
            "tot_p95":   p95(a_totals),
            "tot_max":   max(a_totals),
            "steps_avg": sum(r["steps"] for r in results) / n,
            "tools_avg": sum(r["tools"] for r in results) / n,
            "tok_in":    sum(r["in"] for r in results),
            "tok_out":   sum(r["out"] for r in results),
        }

    def run_bench(label: str, agent_obj) -> dict:
        """Run the concurrent bench against a given agent and print results."""
        global agent
        agent = agent_obj  # one_agent_run / run_concurrent_agents read this
        section(f"5 · CONCURRENT AGENT × {CONC}  ({label})",
                "parallel multi-turn loops with tool calls")
        console.print(f"[dim](warming up {label} agent…)[/]")
        one_agent_run(-1, "Quick check — account 8472193 balance.",
                      {"get_account_balance"})
        results, wall = run_concurrent_agents(CONC)
        console.print(render_agent_table(results))
        agg = aggregate(results, wall)
        accept = agg["perfect"] + agg["extra"]
        console.print(metrics_table([
            ("concurrency",     str(CONC)),
            ("wall clock",      fmt(agg["wall"])),
            ("reply success",
                f"{agg['ok']}/{CONC}  ({100 * agg['ok'] / CONC:.0f}%)"),
            ("tool correctness",
                f"perfect [green]{agg['perfect']}[/]  +extras [yellow]{agg['extra']}[/]  "
                f"partial [yellow]{agg['partial']}[/]  wrong [red]{agg['wrong']}[/]   "
                f"→ acceptable {accept}/{CONC} ({100 * accept / CONC:.0f}%)"),
            ("voice TTFT avg/p50/p95/max",
                f"{fmt(agg['ttft_avg'])} / {fmt(agg['ttft_p50'])} / "
                f"{fmt(agg['ttft_p95'])} / {fmt(agg['ttft_max'])}"),
            ("1st-action avg/p95",
                f"{fmt(agg['act_avg'])} / {fmt(agg['act_p95'])}"),
            ("total avg/p95/max",
                f"{fmt(agg['tot_avg'])} / {fmt(agg['tot_p95'])} / {fmt(agg['tot_max'])}"),
            ("LLM steps avg/call",  f"{agg['steps_avg']:.1f}"),
            ("tool calls avg/call", f"{agg['tools_avg']:.1f}"),
            ("tokens",              f"prompt={agg['tok_in']}  completion={agg['tok_out']}"),
            ("aggregate throughput",
                f"{CONC / agg['wall']:.2f} agents/s   "
                f"{agg['tok_out'] / agg['wall']:.0f} output tok/s"),
        ], title=f"agent aggregate — {label}"))
        return agg

    # Build the two agents we want to compare.
    agent_seq = create_agent(model=llm_tools, tools=TOOLS, system_prompt=SYSTEM_SEQUENTIAL)
    agent_par = create_agent(model=llm_tools, tools=TOOLS, system_prompt=SYSTEM_PARALLEL)

    agg_seq = run_bench("sequential prompt", agent_seq)
    agg_par = run_bench("parallel-tools prompt", agent_par)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SIDE-BY-SIDE comparison
    # ─────────────────────────────────────────────────────────────────────────
    def delta(a: float, b: float) -> str:
        """Format a → b with absolute and percentage change."""
        diff = b - a
        pct = (diff / a * 100) if a else 0.0
        sign = "+" if diff > 0 else ""
        color = "red" if diff > 0 else "green"   # for latency, lower is better
        return f"{fmt(a)} → {fmt(b)}  [{color}]({sign}{pct:.0f}%)[/]"

    section("6 · COMPARISON", "sequential vs parallel-tools system prompt")
    cmp = Table(show_header=True, header_style="bold", border_style="bright_black",
                box=None, pad_edge=False, padding=(0, 2))
    cmp.add_column("metric", style="dim")
    cmp.add_column("sequential")
    cmp.add_column("parallel")
    cmp.add_column("Δ (parallel - sequential)")
    def row(name, a, b, *, lower_better=True, count=False):
        diff = b - a
        pct = (diff / a * 100) if a else 0.0
        sign = "+" if diff > 0 else ""
        good = (diff < 0) if lower_better else (diff > 0)
        color = "green" if good else ("red" if diff else "dim")
        fmt_v = (lambda x: f"{x:.2f}") if count else fmt
        delta_cell = f"[{color}]{sign}{pct:.0f}%[/]" + ("" if count else f"  ({fmt(abs(diff))})")
        cmp.add_row(name, fmt_v(a), fmt_v(b), delta_cell)

    row("voice TTFT avg",   agg_seq["ttft_avg"],  agg_par["ttft_avg"])
    row("voice TTFT p95",   agg_seq["ttft_p95"],  agg_par["ttft_p95"])
    row("total avg",        agg_seq["tot_avg"],   agg_par["tot_avg"])
    row("total p95",        agg_seq["tot_p95"],   agg_par["tot_p95"])
    row("wall clock",       agg_seq["wall"],      agg_par["wall"])
    row("LLM steps avg",    agg_seq["steps_avg"], agg_par["steps_avg"], count=True)
    row("tool calls avg",   agg_seq["tools_avg"], agg_par["tools_avg"], count=True, lower_better=False)
    row("perfect verdicts", agg_seq["perfect"],   agg_par["perfect"],   count=True, lower_better=False)

    console.print(cmp)
    console.print(
        "[dim]Lower latency = green, higher = red. "
        "Higher tool count and more perfect verdicts are good.[/]"
    )
