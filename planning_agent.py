from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunContextWrapper,  ModelSettings, function_tool, ItemHelpers
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from deep_research_system import lead_research_agent
import asyncio
import os

_: bool = load_dotenv(find_dotenv())

gemini_api_key: str | None = os.environ.get("GEMINI_API_KEY")


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
async def ask_user(context: RunContextWrapper, question: str) -> str:
    print(question)
    response = input("Your response: ").strip()
    if response.lower() == 'quit':
        return  # Exit the loop if the user types 'quit'
    else: lines.append(user_input)
    if not response:
        return "User did not provide input."
    return response

planning_agent: Agent = Agent(name="PlanningAgent", 
                          model=special_model,
                          instructions="You are a planning assistant to generate the plan for the research topic and then pass it to the lead research agent.",
                          handoffs=[lead_research_agent],
                          model_settings=ModelSettings(temperature=0.7, max_tokens=1000, max_tool_calls=1)
)

requirement_gathering_agent: Agent = Agent(name="RequirementGatheringAgent", 
                          model=llm_model,
                          instructions="You are a requirement gathering agent that collects user requirements one time. Engage the user with clarifying questions one time only before passing them to the planning agent.",
                          handoffs=[planning_agent],
                          tools=[ask_user],
                          model_settings=ModelSettings(temperature=0.7, max_tokens=1000, max_tool_calls=1),

)
async def call_agent()-> str:


    result = Runner.run_streamed(starting_agent=requirement_gathering_agent, 
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
                print(f"\n\n[{result.last_agent.name} Output]: \n {ItemHelpers.text_message_output(event.item)}\n")
                await asyncio.sleep(0.5)
            else:
                pass  # Ignore other event types
 
    # # ✅ Extract string from agent's result (depends on the SDK version)
    # output = result.final_output
    print("\n\n [Final]: ", result.final_output)
    # # ✅ Make sure it's a string (print it to check format)
    # print("\nAgent Output:\n", output)
    return result.last_agent.name


if __name__ == "__main__":

    lines = []
    user_input = input('What do you want to research today?\n[Input] : ') # The prompt is implicit and appears on a new line each time
    finished_agent = None

    while True:

        if user_input.lower() == 'quit':
            break
        else: lines.append(user_input)
        finished_agent = asyncio.run(call_agent())
        if finished_agent == "ReportWriterAgent":
            print("=== Run complete ===")  
            break

        