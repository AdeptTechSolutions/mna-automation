from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, ManagedAgent

from config import MODEL_API_KEY, MODEL_ID
from prompts import STRATEGY_PROMPT
from tools import human_intervention, save_to_markdown

load_dotenv()


model = LiteLLMModel(
    model_id=MODEL_ID,
    api_key=MODEL_API_KEY,
    temperature=0.0,
)
strategist = CodeAgent(
    tools=[save_to_markdown, human_intervention],
    model=model,
    max_steps=10,
)
managed_strategist = ManagedAgent(
    agent=strategist,
    name="strategist",
    description=STRATEGY_PROMPT,
)

response = strategist.run(
    STRATEGY_PROMPT,
    reset=False,
)
