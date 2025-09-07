import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, RunContextWrapper, ModelSettings, ItemHelpers, ToolsToFinalOutputFunction
from agents.tool import function_tool
from agents.tool_context import ToolContext
import asyncio
from dataclasses import dataclass
from tavily import AsyncTavilyClient
from typing import Literal
from openai import AsyncOpenAI
from typing import List, Optional, Any
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from synthesis_agent import sythesis_agent
import re, json
from enum import Enum

# Define reliability categories
Reliability = Literal["High", "Medium", "Low"]

import os
# Use tavily_client.search() in your agent logic
_: bool = load_dotenv(find_dotenv())
gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")

tavily_api_key: str | None = os.environ.get("TAVILY_API_KEY")

tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", 
    openai_client=external_client,
    )

@function_tool
async def deep_search(wrapper: RunContextWrapper, query: str) -> str:
    """
    Use the deep_search tool to perform deep research based on user's {query} and summarize the deep research when requested.
    spawn multiple searches to accomplish the deep research task.
    """
    result = await tavily_client.search(query=query, search_depth="advanced", max_results=5)
    urls = [item['url'] for item in result.get("results", [])]
    # print("\n\n [Result from deepsearch]: ", result['results'][0]['content'], f"\n\n{urls}\n\n")
    return result


from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

def research_instructions(context: RunContextWrapper, query: str) -> str:
    """
    Dynamically generate research instructions for the web search agent.
    Detect keywords like "deeper" / "summarise" in the user query
    and mutate instructions accordingly.
    """
    user_query = str(context.context).lower()

    # --- base orchestrator instructions with handoff flow ---
    base_instructions = f"""{RECOMMENDED_PROMPT_PREFIX}
    You research work orchestrator.

    Process:
    1. Spwan multiple searches at once to accomplish the deep research task.

    General style:
    - Share findings in concise bullet points.
    - Provide personalised or concrete examples when relevant.
    """

    # --- dynamic mutation ---
    if "deeper" in user_query:
        base_instructions += (
            "\nExtra Requirement: Perform **deeper analysis** by expanding result count and "
            "including multiple perspectives before concluding."
        )
    elif "summarise" in user_query:
        base_instructions += (
            "\nExtra Requirement: Provide a **short summary** with only the key points and "
            "avoid lengthy details."
        )

    return base_instructions


web_search_agent: Agent = Agent(name="WebSearchAgent", 
                          model=llm_model,
                          tools=[deep_search],
                          instructions=research_instructions,
                          model_settings=ModelSettings(temperature=0.1, max_tokens=1000, tool_choice="required"),
                          )




print(f"\n\n [web search_agent]: {web_search_agent.name}, Tools: {[tool.name for tool in web_search_agent.tools]}")

# Step 1: deep search via base
# r0 = Runner.run_sync(starting_agent=web_search_agent, input="Chinese vs German cars, which are better in performance and reliability?")

# # Step 2: facts
# r1 = Runner.run_sync(starting_agent=fact_finding_agent, input=r0.to_input_list())

# # Step 3: citations
# r2 = Runner.run_sync(starting_agent=citation_agent, input=r1.to_input_list())

# # Step 4: credibility
# r3 = Runner.run_sync(starting_agent=source_checker_agent, input=r2.to_input_list())
# print(r3.final_output)
# print("\n\n [Final]: ", r0.final_output)