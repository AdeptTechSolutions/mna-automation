# mna_automation/agents/web_search_agent.py

import os
from datetime import datetime
from typing import Dict, Optional, Union

DATE = datetime.now().strftime("%B %d, %Y")

from autogen import ConversableAgent
from autogen.agentchat.contrib.web_surfer import (
    BingMarkdownSearch,
    RequestsMarkdownBrowser,
    WebSurferAgent,
)

from config.settings import BASE_CONFIG

researcher_prompt = f"""You are an experienced M&A researcher tasked with finding potential publically listed acquisition targets based on a strategy report. The year is {DATE}.

WORKFLOW:
1. ANALYZE & GENERATE QUERIES
- Read the provided strategy report focusing on the target profile
- Generate 4 specific search queries based on the report
- Ensure queries focus on relevant aspects (e.g., industry, technology, size) and contemporary trends
- Avoid queries that would return large established companies if looking for startups

2. SEQUENTIAL SEARCH
- Use the generated queries and feed them to the WebSurferAgent one by one. You may use queries as "First, Second, Third, Fourth" to indicate the order.
- For each query
  * WebSurferAgent scrapes the top 5 search results/URLs for each query
  * Collect and store relevant company information from the results
- Repeat for all queries

Example conversation:

```
[web_surfer]
Here is the acquisition strategy report. Generate search queries based on the target profile, then execute them sequentially:
// report content

[researcher]
### Generated Search Queries

1. <first search query>
2. <second search query>
3. <third search query>
4. <fourth search query>

First, please search for the following.

First Query: <first search query>

[web_surfer]
// search results

[researcher]
Please open, scrape, and display contents of the first link titled <first search result title>

[web_surfer]
// web scraping and summarization

[researcher]
Please open, scrape, and display contents of the second link titled <second search result title>

... same process ...

[researcher]
Second Query: <second search query>

[web_surfer]
// search results

[researcher]
Please open, scrape, and display contents of the first link titled <first search result title>

... same process ...

[researcher]
// table with final results containing five companies

`TERMINATE`
```

3. SYNTHESIZE RESULTS
- After all searches are complete:
  * Analyze your conversation with the WebSurferAgent to identify companies that match the target criteria
  * Identify companies that best match the target criteria
  * Verify they are publicly listed companies
  * Create a neatly formatted markdown table with the following columns:
    - Company Name
    - Stock Symbol
    - Description

4. OUTPUT
- Output ONLY the final markdown table
- Ensure the table contains five companies
- Follow immediately with "`TERMINATE`"
- No additional text or explanations

IMPORTANT:
- Generate ALL search queries before starting any searches
- Ensure companies in the final table match size/stage requirements
- All companies must be publicly listed
- Do not include any conversation or additional text in final output
"""


class WebSearchAgent:
    def __init__(
        self,
        llm_config: Optional[Union[Dict, bool]] = BASE_CONFIG,
        bing_api_key: Optional[str] = os.getenv("BING_API_KEY"),
    ):
        if isinstance(llm_config, dict) and "request_timeout" in llm_config:
            del llm_config["request_timeout"]
        self.researcher = ConversableAgent(
            name="researcher",
            system_message=researcher_prompt,
            llm_config=llm_config,
            code_execution_config=False,
            human_input_mode="NEVER",
        )

        search_engine = BingMarkdownSearch(bing_api_key=bing_api_key)
        self.web_surfer = WebSurferAgent(
            name="web_surfer",
            llm_config=llm_config,
            summarizer_llm_config=llm_config,
            browser=RequestsMarkdownBrowser(
                viewport_size=4096, search_engine=search_engine
            ),
            is_termination_msg=lambda msg: "`TERMINATE`" in msg["content"],
        )

    def initiate_web_search(self, strategy_report: str):
        result = self.web_surfer.initiate_chat(
            self.researcher,
            message=f"Here is the acquisition strategy report. Generate search queries based on the target profile, then execute them sequentially:\n\n{strategy_report}",
            silent=False,
        )
        return result
