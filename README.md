# Multi-Agent Product Launch Manager

## Project Overview
The **Multi-Agent Product Launch System** is a workflow automation framework that simulates a collaborative product launch process using **LangGraph** and **AutoGen**. The system orchestrates multiple specialized agents to produce a structured, end-to-end product launch plan, including market research, strategy planning, risk assessment, marketing content creation, and documentation. It works for any product type, whether physical, software, or service.

## Key Features
- Modular multi-agent workflow for end-to-end product launch planning.
- Structured Markdown outputs including tables, bullet points, and summaries.
- Effort estimation per agent task and overall project phases.
- Conditional loops for risk mitigation and content consistency.
- Works with any product type and is highly customizable.

## Agents and Their Roles

| Agent Name          | Role Description                                                                                                   | Human Input Mode |
|--------------------|-------------------------------------------------------------------------------------------------------------------|----------------|
| Customer Proxy      | Collects product details and requirements from the user.                                                           | ALWAYS         |
| Project Manager     | Oversees workflow, ensures consistency, monitors loops, compiles total effort, and summarizes outputs.           | NEVER          |
| Market Analyst      | Conducts market research, competitor analysis, and customer needs assessment.                                     | NEVER          |
| Strategy Planner    | Converts insights into a detailed product launch roadmap with milestones, timelines, and budgets.                | NEVER          |
| Risk Assessor       | Evaluates feasibility, market, and financial risks, suggests mitigation, and assigns risk scores.               | NEVER          |
| Content Creator     | Generates marketing materials such as ads, blog posts, and social media prompts aligned with strategy.           | NEVER          |
| Documentation Agent | Compiles outputs from all agents into a polished Markdown report summarizing findings, recommendations, and efforts. | NEVER          |

## Prerequisites
- **Python**: 3.8 or higher
- **Required Python packages**:
  - autogen
  - langchain
  - openai
  - pyautogen
  - langgraph

## Installation
1. Clone the repository:
```
git clone https://github.com/Shirish-Master/Multi-Agent-Product-Launch-Manager.git
cd Multi-Agent-Product-Launch-Manager
```
2. Install required packages:
```
pip install -r requirements.txt
```
4. Configure OpenAI API credentials in OAI_CONFIG.json:
```
{
  "model": "gpt-5-mini",
  "api_key": "YOUR_OPENAI_API_KEY",
  "temperature": 0.7
}
```
5. Run the workflow:
```
python ai_product_launch.py
```
6. Enter your product description when prompted. Agents will process inputs and provide a complete product launch report.
