import os
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, ItemHelpers
from agents.run import RunConfig, RunContextWrapper
from agents.tool import function_tool
import asyncio
from dataclasses import dataclass
from tavily import AsyncTavilyClient
from typing import Optional
from pprint import pprint
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

special_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-pro", 
    openai_client=external_client,
    )
@dataclass
class StructuredResponse:
    industry: str                   # e.g., "telco", "finance", "healthcare", "retail"
    technology: Optional[str] = None     # e.g., "churn analysis", "fraud detection", "core KPIs"
    intent: Optional[str] = None 
    research_results: str  = None # e.g., "troubleshooting", "benchmarking", "strategy"

from pydantic import BaseModel

class UserProfile(BaseModel):
    name: str
    city: str
    research_preferences: list

def fetch_user_profile() -> UserProfile:
    # In reality, query DB here
    return UserProfile(
        name="Adeel",
        city="Dubai",
        research_preferences=["AI use cases"]
    )

@function_tool
async def deep_search(wrapper: RunContextWrapper[UserProfile], query: str) -> str:
    """
    Use the deep_search tool to perform deep research based on user's {query} and summarize the deep research when requested.
    """
    result = await tavily_client.search(query=query, max_results=5)
    print ("\n\n [Result]: ", result['results'][0]['content'])
    return result


def research_instructions(context: RunContextWrapper[UserProfile], agent: Agent[UserProfile]) -> str:
    """
    Assess the user's research needs for "deep research" using the deep_search tool
    also provide the summary only when requested
    """
    print("\n\n[Agent Instruction]: ",agent.instructions)

    profile: UserProfile = context.context
    return f"""You’re an AI {agent.name} required to help user name {profile.name}
    who wants to perform deep research on topic {profile.research_preferences[0]}. Share the research in bullets.
    Personalise examples accordingly."""
   

planning_agent: Agent = Agent(name="PlanningAgent", 
                          model=special_model,
                          instructions="You are a planning agent that breaks down complex tasks into simpler steps.",
                          model_settings=ModelSettings(temperature=0.7, max_tokens=1000, tool_choice="none"),
)

orchestrator_agent: Agent = Agent(name="OrchestratorAgent", 
                          instructions="You are an orchestrator assistant. You deletegate tasks to specialized agents using tool call.",
                          model=llm_model,
                          tools=[planning_agent.as_tool(
                              tool_name="planning_agent", 
                              tool_description="A planning agent that uses the scientific reasoning to plan the next step.")],
                          )

agent: Agent = Agent(name="DeepSearchAgent", 
                          # model="gpt-4.1-mini",
                          model=llm_model,
                          tools=[deep_search],
                          instructions=research_instructions,
                          model_settings=ModelSettings(temperature=0.1, max_tokens=1000),
                          )


async def call_agent():


    result = Runner.run_streamed(starting_agent=orchestrator_agent, 
                                input=user_input,
                                context=fetch_user_profile()
                                )
    async for event in result.stream_events():
        if event.type== "delta":
            print(event.delta, end="", flush=True)
                # Delay for visualization
            await asyncio.sleep(0.1)  # 100ms delay between chunks
        # We'll ignore the raw responses event deltas
        elif event.type == "raw_response_event":
            continue
        # When the agent updates, print that
        elif event.type == "agent_updated_stream_event":
            print(f"Agent updated: {event.new_agent.name}")
            continue
        # When items are generated, print them
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- [Tool was called]: ")
            # elif event.item.type == "tool_call_output_item":
            #     print(f"-- Tool output: {json.dumps(event.item.output, indent=2)}")
            elif event.item.type == "message_output_item":
                print(f"\n\n[Deep Research Results]: \n {ItemHelpers.text_message_output(event.item)}")
                await asyncio.sleep(0.5)
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")   
    # # ✅ Extract string from agent's result (depends on the SDK version)
    # output = result.final_output

    # # ✅ Make sure it's a string (print it to check format)
    # print("\nAgent Output:\n", output)



if __name__ == "__main__":

    lines = []
    print("What do you want to research today? \n[Input]: ")
    user_input = "I need real estate inventory management system. Use the best practices to design such system "# The prompt is implicit and appears on a new line each time
    # if user_input.lower() == 'quit':
    #     break  # Exit the loop if the user types 'quit'
    # else: lines.append(user_input)
    asyncio.run(call_agent())
