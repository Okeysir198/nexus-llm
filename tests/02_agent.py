"""Section 3: create_agent — multi-turn agent loop."""
from __future__ import annotations

import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from rich.text import Text

from shared import TOOLS, SYSTEM, llm_tools, section, chat_line, render_message, metrics_table, fmt, console

if __name__ == "__main__":
    USER3 = "Look up balance for account 8472193 and transfer me to billing."
    section("3 · create_agent (streamed)", "token-level multi-turn agent loop")
    console.print(chat_line("USER", USER3))

    agent = create_agent(model=llm_tools, tools=TOOLS, system_prompt=SYSTEM)

    t_start = time.perf_counter()
    ai_steps = 0
    total_in_tokens = 0
    total_out_tokens = 0
    final_reply: AIMessage | None = None
    step_ttft: float | None = None      # TTFT for the current AI step
    in_step = False                     # are we currently accumulating an AI step?

    # Dual stream: "messages" gives per-token chunks; "updates" gives node outputs
    for ev_type, payload in agent.stream(
        {"messages": [HumanMessage(USER3)]},
        stream_mode=["messages", "updates"],
    ):
        elapsed = time.perf_counter() - t_start

        if ev_type == "messages":
            # payload is (message_chunk, metadata)
            chunk, meta = payload
            node = meta.get("langgraph_node", "")
            if node != "model":
                continue
            if not in_step:
                in_step = True
                step_ttft = None
            if step_ttft is None and (chunk.content or getattr(chunk, "tool_call_chunks", None)):
                step_ttft = elapsed
                console.print(Text(f"[{fmt(elapsed):>8}] ", style="dim").append(
                    f"⚡ AI step {ai_steps + 1} TTFT = {fmt(step_ttft)}", style="dim cyan"))
            # vLLM emits usage on the FINAL chunk only when stream_options.include_usage=True.
            # langchain-openai sets that automatically; capture totals here.
            if chunk.usage_metadata:
                total_in_tokens  += chunk.usage_metadata.get("input_tokens", 0)
                total_out_tokens += chunk.usage_metadata.get("output_tokens", 0)

        elif ev_type == "updates":
            # payload is {node_name: state_diff}
            for node, diff in payload.items():
                new_msgs = (diff or {}).get("messages") or []
                for msg in new_msgs:
                    render_message(msg, t_offset=elapsed)
                    if isinstance(msg, AIMessage):
                        ai_steps += 1
                        in_step = False  # this AI step finished
                        if msg.content and not msg.tool_calls:
                            final_reply = msg

    total = time.perf_counter() - t_start
    console.print(metrics_table([
        ("total time",  fmt(total)),
        ("LLM steps",   str(ai_steps)),
        ("tokens",      f"prompt={total_in_tokens}  completion={total_out_tokens}"),
        ("final reply", final_reply.content if final_reply else "[red](none)[/]"),
    ], title="agent summary"))
