from __future__ import annotations

from datetime import datetime
import re


def build_cover_page(topic: str, author: str = "Autonomous Research Agent") -> str:
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    return (
        "# Cover Page\n\n"
        f"**Report Title:** {topic}\n\n"
        f"**Prepared By:** {author}\n\n"
        f"**Generated On:** {now}\n\n"
        "---\n"
    )


def normalize_report_sections(raw_report: str) -> str:
    """Ensure expected section headings exist in a readable order.

    The model is asked to output this format, but we still do a light validation pass.
    """
    def has_heading(section: str) -> bool:
        pattern = rf"^\s{{0,3}}#{{1,6}}\s*{re.escape(section)}\s*$"
        return re.search(pattern, raw_report, flags=re.IGNORECASE | re.MULTILINE) is not None

    required_sections = ["Introduction", "Key Findings", "Challenges", "Future Scope", "Conclusion"]

    # If the core sections are present, preserve content and normalize only the title heading.
    if all(has_heading(section) for section in required_sections):
        if has_heading("Title"):
            return raw_report.strip()

        # Extract first markdown heading as a title candidate when '# Title' is missing.
        first_heading = re.search(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", raw_report, flags=re.MULTILINE)
        title_text = first_heading.group(1).strip() if first_heading else "Autonomous Research Report"
        body = re.sub(r"^\s{0,3}#{1,6}\s+.+?\s*$\n?", "", raw_report, count=1, flags=re.MULTILINE).strip()
        return f"# Title\n\n{title_text}\n\n{body}"

    # Lightweight fallback: wrap content under mandatory structure.
    return (
        "# Title\n\n"
        "Autonomous Research Report\n\n"
        "## Introduction\n\n"
        "This report was generated from autonomous web and knowledge-tool research.\n\n"
        "## Key Findings\n\n"
        f"{raw_report.strip()}\n\n"
        "## Challenges\n\n"
        "Data quality, source reliability, and changing trends remain ongoing challenges.\n\n"
        "## Future Scope\n\n"
        "Further research can add quantitative benchmarking and domain-specific validation.\n\n"
        "## Conclusion\n\n"
        "The topic shows meaningful potential and warrants deeper targeted study."
    )


def assemble_final_report(topic: str, report_body: str) -> str:
    cover = build_cover_page(topic)
    body = normalize_report_sections(report_body)
    return f"{cover}\n\n{body}\n"
