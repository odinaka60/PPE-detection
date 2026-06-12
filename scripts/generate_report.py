"""Runs the report agent over the violation log and write a Markdown report.

    python -m scripts.generate_report
"""
import pathlib
from src.utils.config_loader import load_config
from src.agent.aggregate import load_events, summarise
from src.agent.agent import agent
from langchain.messages import HumanMessage
from src.agent.save import save_report
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main():
    cfg = load_config("configs/config.yaml")

    log_path = pathlib.Path(cfg["audit"]["output_dir"]) / "violations.jsonl"
    events = load_events(str(log_path))
    summary = summarise(events)
    print(summary)

    agent_query = HumanMessage(content=f"Write a complaince report based on this summary {summary} ")
    response = agent.invoke({"messages": [agent_query]})
    save_report(response['messages'][1].content)
  


if __name__ == "__main__":
    main()