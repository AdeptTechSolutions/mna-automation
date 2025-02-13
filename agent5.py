# agent5.py

from pathlib import Path
from typing import Annotated

import autogen
import pandas as pd
from autogen import ConversableAgent, register_function
from configs import OAI_CONFIG
from typing_extensions import Annotated

from prompts import VALUATION_PROMPT
from tools import read_from_markdown, read_json_from_disk, save_to_markdown

LLM_CONFIG = OAI_CONFIG


analyzer = ConversableAgent(
    "analyzer",
    llm_config=LLM_CONFIG,
    system_message=VALUATION_PROMPT,
    human_input_mode="NEVER",
)

executor = ConversableAgent(
    "executor",
    llm_config=False,
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and "TERMINATE" in x["content"],
    default_auto_reply="",
)

register_function(
    read_from_markdown,
    caller=analyzer,
    executor=executor,
    name="read_from_markdown",
    description="Read markdown file content",
)

register_function(
    read_json_from_disk,
    caller=analyzer,
    executor=executor,
    name="read_json_from_disk",
    description="Read JSON data from disk",
)

register_function(
    save_to_markdown,
    caller=analyzer,
    executor=executor,
    name="save_to_markdown",
    description="Save content to markdown file",
)

if __name__ == "__main__":
    analyzer.initiate_chat(
        executor,
        message="Generate comprehensive valuation report analyzing all companies against the acquisition strategy.",
    )
