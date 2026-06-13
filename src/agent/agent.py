from langchain.agents import create_agent
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.rag_tool import search_regulations
from dotenv import load_dotenv
from src.utils.config_loader import load_config

load_dotenv()
cfg = load_config("configs/config.yaml")
model = cfg["agent"]["model"]

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[search_regulations],
)



