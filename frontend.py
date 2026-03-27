from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
OUTPUTS_DIR = ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from main import run_research, save_report  # noqa: E402

DEFAULT_PROVIDER = "groq"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE = 0.2

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Cormorant+Garamond:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #e8edf4;
}

.stApp {
    background:
        radial-gradient(1200px 650px at -10% -20%, rgba(16, 185, 129, 0.22), transparent),
        radial-gradient(1300px 650px at 120% 0%, rgba(59, 130, 246, 0.26), transparent),
        linear-gradient(135deg, #030712 0%, #0b1220 45%, #0a0f1c 100%);
}

.main .block-container {
    max-width: 1040px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    border-radius: 20px;
    padding: 1.8rem 1.8rem;
    background: rgba(8, 14, 28, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.25);
    backdrop-filter: blur(8px);
    box-shadow: 0 12px 36px rgba(2, 6, 23, 0.45);
    animation: fadeUp 450ms ease-out;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0 0 0.6rem 0;
    color: #f8fafc;
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    letter-spacing: -0.3px;
    font-weight: 700;
}

.hero p {
    margin: 0 0 1rem 0;
    color: #cbd5e1;
    line-height: 1.5;
    font-size: 1rem;
}

.credit {
    display: inline-block;
    color: #ffffff;
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 0.5rem 1rem;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 500;
}

.panel {
    border-radius: 16px;
    padding: 1.2rem;
    margin-top: 1.2rem;
    background: rgba(15, 23, 42, 0.64);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.35);
    animation: fadeUp 520ms ease-out;
}

.report-card {
    border-radius: 16px;
    padding: 1.2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    background: rgba(8, 14, 28, 0.82);
    border: 1px solid rgba(125, 211, 252, 0.28);
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.42);
    animation: fadeUp 560ms ease-out;
}

[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label {
    color: #cbd5e1;
    font-weight: 500;
    font-size: 0.95rem;
}

[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: rgba(3, 7, 18, 0.82);
    color: #f8fafc;
    border: 1px solid rgba(125, 211, 252, 0.35);
    border-radius: 10px;
    font-size: 0.95rem;
}

[data-testid="stTextInput"] input::placeholder {
    color: #93a5be;
}

[data-testid="stButton"] button {
    border-radius: 10px;
    border: 1px solid rgba(16, 185, 129, 0.55);
    background: linear-gradient(135deg, #10b981, #14b8a6);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.95rem;
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #2dd4bf, #34d399);
}

[data-testid="stDownloadButton"] button {
    border-radius: 10px;
    border: 1px solid rgba(96, 165, 250, 0.55);
    background: linear-gradient(135deg, #3b82f6, #0ea5e9);
    color: #e8f4ff;
    font-weight: 700;
    font-size: 0.95rem;
}

[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
}

.caption {
    color: #9ab0c9;
    margin-top: 0.25rem;
    font-size: 0.9rem;
}

h2, h3 {
    margin-top: 0.6rem;
    margin-bottom: 0.4rem;
}

.stDivider {
    margin: 0.9rem 0 1.1rem;
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(7px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}
</style>
""",
    unsafe_allow_html=True,
)

load_dotenv(ROOT / ".env")


def list_recent_reports() -> list[Path]:
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        [p for p in OUTPUTS_DIR.glob("report_*.md") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def hydrate_state() -> None:
    if "report_paths" not in st.session_state:
        st.session_state["report_paths"] = list_recent_reports()
    if "current_index" not in st.session_state:
        st.session_state["current_index"] = 0
    if "report" not in st.session_state:
        st.session_state["report"] = ""
    if "output_path" not in st.session_state:
        st.session_state["output_path"] = ""
    if "scroll_top" not in st.session_state:
        st.session_state["scroll_top"] = False


hydrate_state()

if st.session_state["scroll_top"]:
    components.html("<script>window.parent.scrollTo({top: 0, behavior: 'smooth'});</script>", height=0)
    st.session_state["scroll_top"] = False

st.markdown(
    """
<div class="hero">
  <h1>Research Agent</h1>
  <p>Generate comprehensive, structured research reports instantly. Just type your topic and let AI handle the research.</p>
  <div class="credit">✨ Made by Harshit Arora</div>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

paths: list[Path] = st.session_state["report_paths"]
report_names = [p.name for p in paths] if paths else []

if report_names and "selected_report_name" not in st.session_state:
    st.session_state["selected_report_name"] = report_names[0]

selected_name = st.session_state.get("selected_report_name", report_names[0] if report_names else "")
selected_index = report_names.index(selected_name) if report_names else 0

st.subheader("Create Report")
topic = st.text_input(
    "Research Topic",
    value="Impact of AI in Healthcare",
    placeholder="Type your topic...",
)

col_gen, col_info = st.columns([1, 2])
with col_gen:
    run_clicked = st.button("Generate Report", use_container_width=True, type="primary")
with col_info:
    st.caption("New report will be shown at the top automatically.")

if run_clicked:
    if not topic.strip():
        st.error("Please enter a topic.")
    else:
        with st.spinner("Researching and drafting report..."):
            try:
                report = run_research(
                    topic=topic.strip(),
                    provider=DEFAULT_PROVIDER,
                    model=DEFAULT_MODEL,
                    temperature=DEFAULT_TEMPERATURE,
                )
                output_path = save_report(topic.strip(), report, OUTPUTS_DIR)

                st.session_state["report"] = report
                st.session_state["output_path"] = str(output_path)
                st.session_state["report_paths"] = list_recent_reports()
                st.session_state["selected_report_name"] = output_path.name
                st.session_state["scroll_top"] = True
                st.rerun()
            except Exception:
                st.error("Could not generate the report right now. Please try again.")

st.divider()

if paths:
    selected_path = paths[selected_index]
    if selected_path.exists():
        st.session_state["report"] = selected_path.read_text(encoding="utf-8")
        st.session_state["output_path"] = str(selected_path)

if st.session_state["report"]:
    st.subheader("Current Report")
    st.markdown(st.session_state["report"])
    st.download_button(
        label="Download Report",
        data=st.session_state["report"],
        file_name=Path(st.session_state["output_path"]).name if st.session_state["output_path"] else "research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()

st.subheader("Recent Reports")
col1, col2 = st.columns([5, 1])
with col1:
    selected_name = st.selectbox(
        "Recent Reports",
        options=report_names if report_names else ["No reports yet"],
        index=selected_index if report_names else 0,
        disabled=not report_names,
        key="selected_report_name",
        on_change=lambda: st.session_state.update({"scroll_top": True}),
    )
with col2:
    delete_clicked = st.button("Delete", use_container_width=True, disabled=not report_names, key="delete_btn")

if report_names and delete_clicked:
    to_delete = paths[selected_index]
    try:
        to_delete.unlink(missing_ok=True)
        st.session_state["report_paths"] = list_recent_reports()
        updated_paths = st.session_state["report_paths"]
        if not updated_paths:
            st.session_state["report"] = ""
            st.session_state["output_path"] = ""
            st.session_state["selected_report_name"] = ""
        else:
            next_index = min(selected_index, len(updated_paths) - 1)
            st.session_state["selected_report_name"] = updated_paths[next_index].name
            st.session_state["report"] = updated_paths[next_index].read_text(encoding="utf-8")
            st.session_state["output_path"] = str(updated_paths[next_index])
        st.session_state["scroll_top"] = True
        st.rerun()
    except OSError:
        st.error("Could not delete the selected report.")
