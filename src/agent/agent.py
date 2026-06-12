from langchain.agents import create_agent
from src.agent.prompts import SYSTEM_PROMPT
from dotenv import load_dotenv

load_dotenv()

agent = create_agent(
    model="gpt-5-nano",
    system_prompt=SYSTEM_PROMPT,
)



