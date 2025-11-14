from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

# Creates an agent graph that calls tools in a loop until a stopping condition is met.
# visit [Agents](https://docs.langchain.com/oss/python/langchain/agents) documentation
# Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions.
'''
create_agent builds a graph-based agent runtime using LangGraph. 
A graph consists of nodes (steps) and edges (connections) that define how your agent processes information. 
The agent moves through this graph, executing nodes like the model node (which calls the model), the tools node (which executes tools), or middleware.
'''
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 job postings for an ai engineer using langchain in the Rosario, Santa Fe, Argentina area on linkedin and list their details",
                }
            ]
        }
    )

    structured = result.get("structured_response", None)
    print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
