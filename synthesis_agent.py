from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper,  ModelSettings, function_tool, ItemHelpers
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
import os, json,re
from enum import Enum
from typing import List, Any
from urllib.parse import urlparse

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")

tavily_api_key: str | None = os.environ.get("TAVILY_API_KEY")


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


def clean_text(text: str) -> str:
    # Remove things like "Image 7", "Like", "Comment", "Share"
    text = re.sub(r"(Image\s*\d+|Like|Comment|Share|View all.*replies|See more on Facebook)", "", text, flags=re.IGNORECASE)
    
    # Remove excess whitespace and newlines
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    
    return text.strip()

@function_tool
async def build_cited_content(wrapper: RunContextWrapper, result: str) -> str:
    """
    Build cited content from deep search results with inline [n] citations
    and a References section at the end.
 
    """

    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            return "Invalid result format (expected JSON)."
    
    urls: List[str] = [item.get("url", "") for item in result.get("results", [])]
    contents: List[str] = [clean_text(item.get("content", "")) for item in result.get("results", [])]

    # Inline citations
    cited_content_parts = []
    for idx, content in enumerate(contents, start=1):
        if content:  # skip empty entries
            cited_content_parts.append(f"{content} [{idx}]")

    cited_content = "\n\n".join(cited_content_parts)

    # References section
    references = "\n".join([f"[{i+1}] {url}" for i, url in enumerate(urls) if url])

    final_output = f"{cited_content}\n\nReferences:\n{references}" if cited_content else "No relevant content found."

    # print(final_output)  # optional for debugging
    return final_output


class Reliability(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

@function_tool
async def classify_source(context: RunContextWrapper, url: str) -> Reliability:
    """
    Classifies the reliability of a given URL.
    High    -> .edu, .gov, major news outlets
    Medium  -> Wikipedia, industry sites
    Low     -> blogs, forums, unverified sites
    """
    prompt = f"""
    Classify the reliability of this source URL: {url}
    Categories:
    - High: .edu, .gov, or major reputable news outlets
    - Medium: Wikipedia, industry-specific sites, trade publications
    - Low: Blogs, forums, unverified sources
    
    Respond with only one word: High, Medium, or Low.
    """
    domain = urlparse(url).netloc.lower()

    # High reliability: .edu, .gov, major outlets
    if domain.endswith(".edu") or domain.endswith(".gov"):
        return Reliability.HIGH
    if any(outlet in domain for outlet in ["bbc.com", "cnn.com", "nytimes.com", "reuters.com", "apnews.com"]):
        return Reliability.HIGH

    # Medium reliability: Wikipedia, industry sites
    if "wikipedia.org" in domain:
        return Reliability.Medium
    if any(industry in domain for industry in ["forbes.com", "techcrunch.com", "nature.com"]):
        return Reliability.MEDIUM
    # Default: Low reliability (blogs, forums, unknown sites)
    return Reliability.LOW

@function_tool
async def fact_finder(context: RunContextWrapper, results: str) -> str:
    docstring = """
    Reflects and verifies whether the extracted results are based on the factual information.
    Uses reasoning + available knowledge. spawn multiple instances to validate each fact
    Highlight facts and disagreements in your response
    """
    prompt = f"""
    You are a fact-finding agent. Verify the following answer with accurate and trustworthy information.
    [Context]: {context.context}
    [Results]: {results}

    - If it's factual, return the verified fact.
    - If uncertain, state that clearly and suggest where it could be verified.
    - Keep the response concise and reliable.
    """

    return prompt
   # Call the LLM via context (standard Agent SDK pattern)


def dynamic_instructions(context: RunContextWrapper, result: str) -> str:
    """
    Dynamically generate synthesis agent instructions.
    """
    return f"""
    You are a synthesis assistant. You Collect the research work from multiple members of research team:

    1.FactFindingAgent
    2.CitationAgent
    3.SourceCheckerAgent
    
    You synthesize the outcome of the research team into organized insights.

    Process:
    1. Call the **FactFindingAgent** tool to get reflection on the facts gathered during the research work.
    2. Call the **CitationAgent** tool to add inline [n] citations and build a References section.
    3. Call the **SourceCheckerAgent** tool to assess credibility.

    General style:
    - Gather key insights from all the team members.
    - Ensure all facts are verified and cited properly.
    - Highlight any disagreements or uncertainties in the findings.
    """



sythesis_agent: Agent = Agent(name="SythesisAgent",
                                instructions=dynamic_instructions,
                                model=llm_model,
                                model_settings=ModelSettings(tool_choice="required"),
                                )                       

fact_finding_agent: Agent = sythesis_agent.clone(name="FactFindingAgent",
                                             instructions="You are a fact-finder assistant. You reflect the factual information and Spot Disagreements.",
                                             tools=[fact_finder],
                                             model_settings=ModelSettings(temperature=0.1,  tool_choice="required"),
                                             )

citation_agent: Agent = sythesis_agent.clone(name="CitationAgent",
                                             instructions="You are a citation assistant. You provide inline [n] citations & references against the facts received in the input.",
                                             tools=[build_cited_content],
                                             model_settings=ModelSettings(temperature=0.1, tool_choice="required")
                                             )


source_checker_agent: Agent = sythesis_agent.clone(name="SourceCheckerAgent",
                                             instructions="You are a source-checker assistant. You check the credibility of sources using web search.",
                                             tools=[classify_source],
                                             model_settings=ModelSettings(temperature=0.1, tool_choice="required"))

sythesis_agent.tools.append(fact_finding_agent.as_tool(
    tool_name="FactFindingAgent",
    tool_description="Use this agent to find and verify facts related to the research topic.",),
)
sythesis_agent.tools.append(citation_agent.as_tool(
    tool_name="CitationAgent",
    tool_description="Use this agent to add inline citations and a References section to the verified facts.",),
)
sythesis_agent.tools.append(source_checker_agent.as_tool(
    tool_name="SourceCheckerAgent",
    tool_description="Use this agent to assess the credibility of sources and provide a reliability rating.",))


print(f"\n\n [web search_agent]: {sythesis_agent.tools}, Tools: {[tool.name for tool in sythesis_agent.tools]}")
