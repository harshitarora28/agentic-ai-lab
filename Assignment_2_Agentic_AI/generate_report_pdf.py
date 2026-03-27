#!/usr/bin/env python3
"""
Generate a comprehensive project report as PDF.
Uses reportlab to create a professional report document.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime

def create_report_pdf():
    filename = "Project_Report_Assignment_2.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=0.4*inch, bottomMargin=0.4*inch)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=6,
        spaceBefore=3,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=13
    )
    
    # ===== COVER PAGE =====
    story.append(Spacer(1, 0.4*inch))
    story.append(Paragraph("Autonomous Research Agent", title_style))
    story.append(Spacer(1, 0.15*inch))
    
    cover_info = [
        ["Project Name", "Autonomous Research Agent using LangChain"],
        ["Submitted By", "Harshit Arora"],
        ["Date", datetime.now().strftime("%B %d, %Y")],
        ["Technology", "Python, LangChain, Streamlit, LLMs"],
    ]
    
    cover_table = Table(cover_info, colWidths=[2*inch, 3.5*inch])
    cover_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    story.append(cover_table)
    story.append(PageBreak())
    
    # ===== TITLE PAGE =====
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Autonomous Research Agent", title_style))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph("A ReAct-based agent for autonomous research and report generation", 
                          ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=13, alignment=TA_CENTER, textColor=colors.grey)))
    story.append(Spacer(1, 0.2*inch))
    
    # ===== INTRODUCTION =====
    story.append(Paragraph("1. Introduction", heading_style))
    intro_text = """
    In today's digital world, the amount of information available on the internet is massive and continuously growing. 
    Manually researching a topic requires significant time and effort, as it involves searching across multiple sources, 
    filtering useful data, understanding it, and then organizing it into a meaningful format.
    <br/><br/>
    This project focuses on building an Autonomous Research Agent that automates this entire process. The agent is designed 
    using LangChain and powered by a Large Language Model (LLM) through the Groq API. It mimics human-like research behavior 
    by collecting data, analyzing it, and generating structured reports.
    <br/><br/>
    The system reduces manual effort and increases efficiency by providing quick, organized, and relevant insights on any 
    given topic. It demonstrates how AI agents can be used in real-world scenarios such as academic research, business analysis, 
    and content creation.
    """
    story.append(Paragraph(intro_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    # ===== PROBLEM STATEMENT =====
    story.append(Paragraph("2. Problem Statement", heading_style))
    problem_text = """
    Traditional research methods are time-consuming and often inefficient. Users need to:
    <br/><br/>
    • Search multiple websites<br/>
    • Read large amounts of content<br/>
    • Extract important points manually<br/>
    • Organize information into a report<br/>
    <br/>
    This process is repetitive and prone to missing important insights.
    <br/><br/>
    <b>The goal of this project is to solve this problem by creating an intelligent agent that:</b><br/>
    • Automatically searches for information<br/>
    • Understands and processes the data<br/>
    • Extracts key insights<br/>
    • Generates a well-structured report<br/>
    """
    story.append(Paragraph(problem_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    # ===== OBJECTIVES =====
    story.append(Paragraph("3. Objectives", heading_style))
    objectives_text = """
    The main objectives of this project are:
    <br/><br/>
    • To design an AI agent capable of autonomous research<br/>
    • To integrate multiple tools using LangChain<br/>
    • To use an LLM for understanding and generating content<br/>
    • To automate the process of report generation<br/>
    • To provide structured and meaningful outputs<br/>
    """
    story.append(Paragraph(objectives_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    # ===== SYSTEM ARCHITECTURE =====
    story.append(Paragraph("4. System Architecture Overview", heading_style))
    arch_text = """
    The system is built using a modular architecture where different components work together:
    <br/><br/>
    <b>User Input Module:</b> Accepts the research topic from the user<br/>
    <b>Agent (Core Controller):</b> Decides which tool to use, manages the workflow using the ReAct approach<br/>
    <b>Web Search Tool:</b> Fetches real-time information from the internet<br/>
    <b>Knowledge Tool (Wikipedia):</b> Provides reliable and detailed information<br/>
    <b>LLM (Groq API):</b> Processes the collected data, generates summaries and reports<br/>
    <b>Output Generator:</b> Organizes content into structured format<br/>
    """
    story.append(Paragraph(arch_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    # ===== WORKING OF THE SYSTEM =====
    story.append(Paragraph("5. Working of the System", heading_style))
    working_text = """
    The Autonomous Research Agent works in multiple steps:
    <br/><br/>
    <b>Step 1: Input Collection</b><br/>
    The user provides a topic such as "Impact of AI in Healthcare". This acts as the starting point for the research process.
    <br/><br/>
    <b>Step 2: Understanding the Query</b><br/>
    The LLM analyzes the topic and identifies what type of information is required, such as definitions, impacts, challenges, 
    and future trends.
    <br/><br/>
    <b>Step 3: Information Retrieval</b><br/>
    The agent uses:<br/>
    • Web Search Tool for latest and diverse information<br/>
    • Wikipedia Tool for structured and reliable knowledge<br/>
    This ensures both breadth and depth in the research.
    <br/><br/>
    <b>Step 4: Data Processing</b><br/>
    The collected data is passed to the LLM, which:<br/>
    • Removes irrelevant information<br/>
    • Extracts key insights<br/>
    • Summarizes large content into concise points<br/>
    <br/>
    <b>Step 5: Content Organization</b><br/>
    The agent organizes the processed data into sections like:<br/>
    • Introduction<br/>
    • Key Findings<br/>
    • Challenges<br/>
    • Future Scope<br/>
    • Conclusion<br/>
    <br/>
    <b>Step 6: Report Generation</b><br/>
    Finally, a complete and structured report is generated, ready for presentation or submission.
    """
    story.append(Paragraph(working_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    story.append(PageBreak())
    story.append(Paragraph("6. Agent Design (ReAct Approach)", heading_style))
    agent_text = """
    This project uses the ReAct (Reasoning + Acting) framework, which allows the agent to behave more intelligently.
    <br/><br/>
    <b>Reasoning:</b> The agent thinks about what action to take next<br/>
    <b>Acting:</b> It uses tools like web search or Wikipedia<br/>
    <b>Observation:</b> It collects results and updates its understanding<br/>
    <br/>
    This cycle continues until the agent gathers enough information to generate the final report.
    <br/><br/>
    This approach makes the system dynamic, flexible, and closer to human decision-making.
    """
    story.append(Paragraph(agent_text, body_style))
    story.append(Spacer(1, 0.1*inch))
    
    # ===== ADVANTAGES =====
    story.append(Paragraph("7. Advantages of the System", heading_style))
    advantages_text = """
    • Saves time by automating research<br/>
    • Reduces manual effort<br/>
    • Provides structured and organized output<br/>
    • Combines multiple information sources<br/>
    • Improves accuracy using AI understanding<br/>
    """
    story.append(Paragraph(advantages_text, body_style))
    story.append(Spacer(1, 0.08*inch))
    
    # ===== LIMITATIONS =====
    story.append(Paragraph("8. Limitations", heading_style))
    limitations_text = """
    • Depends on the quality of available data<br/>
    • May generate slightly generalized responses<br/>
    • Requires internet access for web tools<br/>
    • Accuracy depends on the LLM<br/>
    """
    story.append(Paragraph(limitations_text, body_style))
    
    # Build PDF
    doc.build(story)
    print(f"✅ Report generated successfully: {filename}")

if __name__ == "__main__":
    create_report_pdf()
