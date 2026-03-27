# Assignment 2: Autonomous Research Agent

A LangChain-based autonomous research agent that searches the web, extracts knowledge, and generates a structured research report. It supports multiple LLM providers and includes a Streamlit UI plus a CLI.

## Features

- ReAct-style agent with tool use (web search + Wikipedia)
- Supports OpenAI, Anthropic, and Groq providers
- Generates consistent, structured markdown reports
- Streamlit UI for one-click report generation
- CLI interface for automation and testing

## Tech Stack

- Python 3.12+
- LangChain + langchain_classic (ReAct agent)
- DuckDuckGo Search + Wikipedia tools
- Streamlit UI

## Architecture (High Level)

1. User provides a topic (UI or CLI)
2. ReAct agent invokes tools to search and gather evidence
3. Findings are normalized into a structured report
4. Output is saved as markdown in `outputs/`

## Project Structure

```
Assignment_2_Agentic_AI/
├─ frontend.py
├─ src/
│  ├─ main.py
│  └─ report_formatter.py
├─ outputs/
├─ docs/
│  └─ samples/
├─ .env.example
├─ requirements.txt
└─ README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set your API key.

## Environment Variables

Set only the provider you plan to use:

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
```

## Run (Frontend)

```bash
streamlit run frontend.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Run (CLI)

### OpenAI

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider openai --model gpt-4o-mini
```

### Anthropic

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider anthropic --model claude-3-5-sonnet-latest
```

### Groq

```bash
python src/main.py --topic "Impact of AI in Healthcare" --provider groq --model llama-3.3-70b-versatile
```

## Output Format

Reports are saved as markdown in `outputs/` and include:

- Cover Page
- Title
- Introduction
- Key Findings
- Challenges
- Future Scope
- Conclusion

## Sample Outputs

Sample reports are stored in `docs/samples/`.

## Troubleshooting

- If a provider fails, verify the correct API key is set in `.env`.
- If the UI is not loading, re-run `streamlit run frontend.py`.
- If you see tool errors, reinstall dependencies from `requirements.txt`.

## Submission Notes

- Upload the full repository to GitHub.
- Include sample reports from `docs/samples/`.
