import os
import json
from typing import Dict, Any
from langgraph.graph import Graph
import autogen

# Load configuration from OAI_CONFIG.json
with open("OAI_CONFIG.json", "r") as config_file:
    config = json.load(config_file)

# Initialize the language model configuration for AutoGen
llm_config = {
    "config_list": [
        {
            "model": config["model"],
            "api_key": config["api_key"],
            "temperature": config["temperature"]
        }
    ]
}

# ANSI escape codes for colored output
GREEN = "\033[92;1m"
BLUE_BOLD = "\033[94;1m"
RESET = "\033[0m"

# Define agent roles and their system messages
AGENT_CONFIGS = {
    "Customer_Proxy": {
        "system_message": """Acts as the customer’s representative. Collects and conveys detailed product requirements, including features, target audience, goals, constraints, and preferred marketing tone. Provides examples and references to similar products if needed. Ensures Project Manager receives all context necessary to start the workflow.""",
        "human_input_mode": "ALWAYS"
    },
    "Project_Manager": {
        "system_message": """Coordinates the entire product launch workflow. Receives detailed requirements from the Customer_Proxy, manages inter-agent communication, resolves conflicts, detects excessive loops, and ensures consistency and completeness of outputs. Summarizes insights, roadmap, risks, and content for the Documentation Agent in structured Markdown. Tracks and aggregates estimated effort per phase and ensures deadlines are realistic.""",
        "human_input_mode": "NEVER"
    },
    "Market_Analyst": {
        "system_message": """Conducts in-depth market research for the product. Produces a comprehensive insights report covering: market size, segmentation, target audience demographics, competitor analysis, consumer preferences, pricing trends, and emerging market opportunities. Outputs must be structured in Markdown, with tables, bullet points, and references where applicable. Estimate effort: 2 days per major insight category. Highlight potential risks or gaps in market understanding.""",
        "human_input_mode": "NEVER"
    },
    "Strategy_Planner": {
        "system_message": """Transforms the Market Analyst’s insights into a detailed product launch roadmap. Includes clear milestones, timelines, budgets, resource allocation, risk mitigation plans, and dependencies. Produces structured Markdown outputs with tables and charts. If Risk_Assessor flags risks >7, refine roadmap to mitigate them. Estimate effort: 3 days per major milestone, budget, or resource allocation category.""",
        "human_input_mode": "NEVER"
    },
    "Risk_Assessor": {
        "system_message": """Analyzes the Strategy Planner's roadmap to identify potential risks in feasibility, market competition, operational execution, and budget. Assigns risk scores (1-10) per category, highlights high-risk items, and provides mitigation suggestions. Structured Markdown output should include a risk table, recommendations, and potential impact. If risk >7, recommend refinement. Estimate effort: 1 day per risk category.""",
        "human_input_mode": "NEVER"
    },
    "Content_Creator": {
        "system_message": """Creates marketing collateral aligned with the product roadmap and market insights. Produces ads, blog posts, social media campaigns, and promotional text in Markdown format. Ensures messaging is consistent with strategy, target audience, and brand tone. Adjusts content iteratively if Project Manager identifies inconsistencies. Estimate effort: 2 days per content piece. Include content outlines, sample copy, and distribution channels.""",
        "human_input_mode": "NEVER"
    },
    "Documentation_Agent": {
        "system_message": """Aggregates outputs from all agents, including market insights, roadmap, risk analysis, and marketing content. Produces a polished, structured Markdown report with tables, headings, and summaries. Includes estimated effort per phase and iteration notes. Ensures all information is clear, comprehensive, and ready for stakeholder review. Estimate effort: 2 days per major section.""",
        "human_input_mode": "NEVER"
    }
}


# Conversational messages
MESSAGES = {
    "customer_to_project_manager": """
Please initiate the product launch process for the product described by the customer. Collect all relevant details, including product features, target audience, launch goals, budget constraints, and preferred marketing tone or style. Coordinate with the Market Analyst to start detailed research and ensure all agents follow the workflow to produce structured, consistent, and actionable outputs.
""",
    "project_manager_to_market_analyst": """
The customer has provided detailed requirements for a product launch. Conduct comprehensive market research covering: market size, segmentation, target audience demographics, competitor analysis, consumer preferences, pricing trends, and potential gaps or opportunities. Produce a structured Markdown report with tables, bullet points, and references where applicable. Include estimated effort per insight category (2 days each) and highlight any potential risks or uncertainties.
""",
    "market_analyst_to_project_manager": """
I have completed the comprehensive market insights report, including detailed tables, bullet points, references, and effort estimates. Please review and pass it to the Strategy Planner to create a detailed product launch roadmap.
""",
    "project_manager_to_strategy_planner": """
The Market Analyst has provided a detailed insights report. Please create a product launch roadmap including clear milestones, timelines, budgets, resource allocation, and dependencies. Produce structured Markdown outputs with tables and charts. Include estimated effort per milestone, budget, or resource allocation category (3 days each). If risks are identified later, refine the roadmap accordingly. Report back to me.
""",
    "strategy_planner_to_project_manager": """
I have completed the detailed product launch roadmap, including milestones, timelines, budgets, and resource allocations, with effort estimates. Please pass it to the Risk Assessor for evaluation.
""",
    "project_manager_to_risk_assessor": """
The Strategy Planner has provided the detailed roadmap. Please evaluate potential risks in feasibility, market competition, operational execution, and budget constraints. Assign risk scores (1-10) per category and provide mitigation suggestions. Present your findings in a structured Markdown report with a risk table, recommendations, and potential impact. If any risk exceeds 7, indicate that roadmap refinement is needed. Estimate effort: 1 day per risk category. Report back to me.
""",
    "risk_assessor_to_project_manager": """
I have completed the risk assessment report, including risk scores, mitigation suggestions, and structured tables. If any risk score is above 7, the roadmap requires refinement by the Strategy Planner. Otherwise, please proceed to the Content Creator.
""",
    "project_manager_to_content_creator": """
The roadmap has been approved with acceptable risk levels. Please generate marketing materials, including ads, blog posts, social media prompts, or other promotional content, based on the Market Analyst's insights and the Strategy Planner's roadmap. Produce structured Markdown outputs with content outlines, sample copy, and suggested distribution channels. Ensure messaging is consistent with the strategy and target audience. Estimate effort: 2 days per content piece. Adjust content iteratively if inconsistencies are found. Report back to me.
""",
    "content_creator_to_project_manager": """
I have completed the marketing materials, including content outlines, sample copy, and distribution channel suggestions, with effort estimates. Please review for consistency with the strategy and market insights. If adjustments are needed, I will revise accordingly.
""",
    "project_manager_to_documentation_agent": """
All outputs (market insights, roadmap, risk assessment, marketing materials) are ready. Please compile them into a polished, structured Markdown report summarizing all findings, effort estimates per phase, and iteration notes. Include tables, headings, and summaries for clarity. Report back to me with the final launch report.
""",
    "documentation_agent_to_project_manager": """
I have compiled the final product launch report, including structured summaries of all agent outputs, tables, and effort estimates. Please review to finalize the project.
""",
    "project_manager_to_customer_proxy": """
Final product launch report with total effort estimates in days for all phases, including market research, roadmap planning, risk assessment, and marketing content.
"""
}



# Initialize AutoGen agents
agents = {
    name: autogen.ConversableAgent(
        name=name,
        system_message=config["system_message"],
        code_execution_config=False,
        llm_config=llm_config,
        human_input_mode=config["human_input_mode"]
    ) for name, config in AGENT_CONFIGS.items()
}

# Function to create a LangGraph node using an AutoGen agent
def create_agent_node(agent_name: str):
    def node_function(state: Dict[str, Any]) -> Dict[str, Any]:
        # Get the agent and relevant message
        agent = agents[agent_name]
        previous_output = state.get("context", "")
        agent_input = state.get("input", "")
        
        # Determine the appropriate message based on the agent and state
        if agent_name == "Customer_Proxy":
            message = agent_input  # User-provided product description
        elif agent_name == "Project_Manager":
            if state.get("last_agent") == "Customer_Proxy":
                message = MESSAGES["customer_to_project_manager"]
            elif state.get("last_agent") == "Market_Analyst":
                message = MESSAGES["project_manager_to_strategy_planner"]
            elif state.get("last_agent") == "Strategy_Planner":
                message = MESSAGES["project_manager_to_risk_assessor"]
            elif state.get("last_agent") == "Risk_Assessor":
                message = MESSAGES["project_manager_to_content_creator"]
            elif state.get("last_agent") == "Content_Creator":
                message = MESSAGES["project_manager_to_documentation_agent"]
            elif state.get("last_agent") == "Documentation_Agent":
                message = MESSAGES["project_manager_to_customer_proxy"]
            else:
                message = agent_input
        else:
            message_key = f"{state.get('last_agent', '').lower()}_to_{agent_name.lower()}"
            message = MESSAGES.get(message_key, agent_input)

        # Simulate conversation with the agent
        response = agent.generate_reply(messages=[{"content": f"{previous_output}\n\n{message}", "role": "user"}])
        
        # Update state
        state["context"] = previous_output + "\n" + response
        state["output"] = response
        state["last_agent"] = agent_name
        
        # Update loop counts for Risk Assessor and Content Creator
        if agent_name == "Risk_Assessor" and "score > 7" in response.lower():
            state["risk_loop_count"] = state.get("risk_loop_count", 0) + 1
        elif agent_name == "Content_Creator" and "inconsistent" in response.lower():
            state["content_loop_count"] = state.get("content_loop_count", 0) + 1
            
        return state
    
    return node_function

# Create LangGraph workflow
graph = Graph()

# Add nodes for each agent
for agent_name in AGENT_CONFIGS.keys():
    graph.add_node(agent_name, create_agent_node(agent_name))

# Define edges
graph.add_edge("Customer_Proxy", "Project_Manager")
graph.add_edge("Project_Manager", "Market_Analyst")
graph.add_edge("Market_Analyst", "Project_Manager")

# Conditional edges for loops
graph.add_conditional_edges(
    "Project_Manager",
    lambda state: (
        "Strategy_Planner" if state.get("last_agent") == "Market_Analyst" else
        "Risk_Assessor" if state.get("last_agent") == "Strategy_Planner" else
        "Content_Creator" if state.get("last_agent") == "Risk_Assessor" else
        "Documentation_Agent" if state.get("last_agent") == "Content_Creator" else
        "Customer_Proxy" if state.get("last_agent") == "Documentation_Agent" else
        "Customer_Proxy"
    ),
    {
        "Strategy_Planner": "Strategy_Planner",
        "Risk_Assessor": "Risk_Assessor",
        "Content_Creator": "Content_Creator",
        "Documentation_Agent": "Documentation_Agent",
        "Customer_Proxy": "Customer_Proxy"
    }
)

graph.add_edge("Strategy_Planner", "Project_Manager")
graph.add_conditional_edges(
    "Risk_Assessor",
    lambda state: (
        "Project_Manager" if state.get("risk_loop_count", 0) < 3 and "score > 7" in state.get("output", "").lower()
        else "Project_Manager"
    ),
    {
        "Project_Manager": "Project_Manager"
    }
)
graph.add_conditional_edges(
    "Content_Creator",
    lambda state: (
        "Project_Manager" if state.get("content_loop_count", 0) < 3 and "inconsistent" in state.get("output", "").lower()
        else "Project_Manager"
    ),
    {
        "Project_Manager": "Project_Manager"
    }
)
graph.add_edge("Documentation_Agent", "Project_Manager")

# Set entry point
graph.set_entry_point("Customer_Proxy")

# Compile the graph
app = graph.compile()

# Start the conversation by prompting the user for input
print("""Please provide a detailed product description for your AI-powered product launch. Include:
1. Product Name and Type
2. Key Features and Technical Specifications
3. Unique Selling Points and Advantages over Competitors
4. Target Audience and Demographics
5. Launch Goals, Timeline, and Budget Constraints
6. Desired Marketing Tone, Style, or Themes
7. References to similar products or industry trends""")
user_input = input()

# Initialize state
state = {
    "input": user_input,
    "context": "",
    "last_agent": "",
    "output": "",
    "risk_loop_count": 0,
    "content_loop_count": 0
}

# Run the graph
for output in app.stream(state):
    for agent_name, agent_state in output.items():
        print(f"{BLUE_BOLD}Agent: {agent_name}{RESET}")
        print(f"{GREEN}Output: {agent_state['output']}{RESET}\n")
        state.update(agent_state)

print("Project completed.")