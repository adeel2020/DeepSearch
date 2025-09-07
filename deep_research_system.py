import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, ItemHelpers, ModelSettings
from agents.run import RunConfig, RunContextWrapper
from agents.tool import function_tool

from research_agents import web_search_agent
from synthesis_agent import sythesis_agent
from report_writer import report_writer_agent
from dataclasses import dataclass
from tavily import AsyncTavilyClient
from typing import Optional
from pprint import pprint

import os
import asyncio


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
    model="gemini-2.5-pro", 
    openai_client=external_client,
    )

lead_research_agent: Agent = Agent(name="LeadResearchAgent", 
                            instructions="""
                            You are an orchestrator assistant. You manage specialized agent to research and report the findings.
                            Your main goal is call specialize tool for designated tasks.

                            Follow these structured steps for each deep research task:

                            Step 1. Assign the deep research task to web search agent and Collect the research work before assigning it to synthesis agent. 
                            Step 2. Assign the synthesis task to synthesis agent to combine the research work into organized insights.
                            Step 3. Assign the writing task to report writer agent to write a detailed and comprehensive professional report based on the feedback from synthesis agent.
                            Step 4. Inline citation and reference are included in the report writer agent's response

                            Always reflect on the output of writer to avoid hallucination before providing the final output to user.""",
                            model=llm_model,
                            # handoffs=[web_search_agent, report_writer_agent],
                          tools=[
                            web_search_agent.as_tool(
                                    tool_name="ResearchAgent", 
                                    tool_description="Use this agent to perform deep research on user's topic using web search.",),
                            sythesis_agent.as_tool(
                                    tool_name="SynthesisAgent", 
                                    tool_description="Assign the synthesis task to synthesis agent to combine the research work into organized insights",),
                            report_writer_agent.as_tool(
                                    tool_name="ReportWriterAgent",
                                    tool_description="Writes the professional reports after the research synthesise is available without missing any details.",)]                            
                          )


        


        
