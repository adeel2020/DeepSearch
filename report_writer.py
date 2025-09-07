from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper,  ModelSettings
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
import os

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

report_writer_agent: Agent = Agent(name="ReportWriterAgent",
                                instructions="You are a report writer assistant. You write comprehensive professional reports based on the research provided by lead research agent." \
                                " Add inline citations [n] and references in the report where applicable. All sections should be well-structured with headings and subheadings. " \
                                " Use formal tone and professional language. Ensure clarity and coherence throughout the report. Use bullet points and numbered lists where appropriate.",    
                                model=llm_model,
                                )                       