"""Minimal LangChain ReAct agent + Satsignal audit trail.

Mirrors the reviewer cold-start flow: policy snapshot at chain start,
one commit-reveal anchor per ReAct step, evidence-bundle manifest at
chain end. Run with a real API key against your matter to see real txids
in your Satsignal workspace.

    pip install langchain langchain-openai langchain-satsignal
    export SATSIGNAL_API_KEY="sk_live_..."
    export OPENAI_API_KEY="sk-..."
    python examples/react_agent.py
"""
import os

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langchain_satsignal import SatsignalCallbackHandler, SatsignalConfig


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Demo tool — not safe for prod."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


@tool
def echo(text: str) -> str:
    """Echo the input back. Stand-in for any side-effecting tool."""
    return text


PROMPT = PromptTemplate.from_template(
    "Answer using these tools: {tools}\n"
    "Tool names: {tool_names}\n\n"
    "Question: {input}\n\n"
    "{agent_scratchpad}"
)


def main() -> None:
    api_key = os.environ.get("SATSIGNAL_API_KEY")
    if not api_key:
        raise SystemExit("Set SATSIGNAL_API_KEY")

    handler = SatsignalCallbackHandler(SatsignalConfig(
        api_key=api_key,
        matter_slug="agent-runs",
        agent_name="react-demo",
        agent_version="0.1.0",
        # decide_on="agent_action" — default; one anchor per ReAct step.
        # fail_open=True            — default; agent runs even if Satsignal blips.
    ))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [calculator, echo]
    agent = create_react_agent(llm, tools, PROMPT)
    executor = AgentExecutor(
        agent=agent, tools=tools,
        verbose=True, handle_parsing_errors=True, max_iterations=5,
    )

    result = executor.invoke(
        {"input": "What is (12*4) + 7? Echo the result."},
        config={"callbacks": [handler]},
    )
    print(f"\nAgent result: {result['output']}")


if __name__ == "__main__":
    main()
