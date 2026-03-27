from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

try:
    from report_formatter import assemble_final_report
except ImportError:
    # Supports imports when main.py is loaded as src.main from frontend apps.
    from src.report_formatter import assemble_final_report


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"[\s-]+", "-", value)
    return value[:80] if value else "topic"


def get_llm(provider: str, model: str, temperature: float):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    raise ValueError("Unsupported provider. Use 'openai', 'anthropic', or 'groq'.")


def build_tools() -> list[Tool]:
    web_tool = DuckDuckGoSearchRun(name="web_search")

    wiki_api = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=2500)
    wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)

    return [
        Tool(
            name="WebSearch",
            func=web_tool.run,
            description="Search the web for recent and relevant information.",
        ),
        Tool(
            name="WikipediaKnowledge",
            func=wiki_tool.run,
            description="Fetch trusted encyclopedic background knowledge from Wikipedia.",
        ),
    ]


def build_react_prompt() -> PromptTemplate:
    template = """
You are an Autonomous Research Agent using ReAct style reasoning.

You must:
1) Research the input topic using available tools.
2) Cross-check claims across multiple sources.
3) Produce a detailed and structured final report in markdown.

Available tools:
{tools}

Use this format exactly while reasoning:
Question: the input topic
Thought: your reasoning step
Action: one of [{tool_names}]
Action Input: the search query for the tool
Observation: result of the tool call
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough information
Final Answer: provide only the final markdown report

The Final Answer report MUST include these sections:
# Title
## Introduction
## Key Findings
## Challenges
## Future Scope
## Conclusion

Question: {input}
{agent_scratchpad}
""".strip()

    return PromptTemplate.from_template(template)


def create_agent_executor(provider: str, model: str, temperature: float) -> AgentExecutor:
    llm = get_llm(provider, model, temperature)
    tools = build_tools()
    prompt = build_react_prompt()

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )


def run_research(topic: str, provider: str, model: str, temperature: float) -> str:
    executor = create_agent_executor(provider, model, temperature)
    result = executor.invoke({"input": topic})

    output = result.get("output")
    if not output:
        raise RuntimeError("Agent did not produce an output.")

    return assemble_final_report(topic=topic, report_body=output)


def save_report(topic: str, report: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{slugify(topic)}_{ts}.md"
    output_path = out_dir / filename
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Research Agent (LangChain)")
    parser.add_argument("--topic", required=True, help="Research topic")
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "anthropic", "groq"],
        help="LLM provider",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name. Example: gpt-4o-mini, claude-3-5-sonnet-latest, llama-3.3-70b-versatile",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    # Basic key checks to fail fast with a clear message.
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Add it to .env")
    if args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError("ANTHROPIC_API_KEY not set. Add it to .env")
    if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError("GROQ_API_KEY not set. Add it to .env")

    report = run_research(
        topic=args.topic,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )

    output_path = save_report(args.topic, report, Path(args.out_dir))
    print("\n=== FINAL REPORT GENERATED ===")
    print(report)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
