# %%
import langchain
from dotenv import load_dotenv

load_dotenv()


# %%
print(langchain.__version__)
#
# %%
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.tools import Tool

# %%
# llm = ChatOpenAI(model="gpt-5-mini")
llm = ChatOpenAI(model="gpt-4o")
# %%
prompt = hub.pull("hwchase17/react")
print(prompt)

# %%
def get_word_length(word: str) -> int:
    return len(word)

# %%
tools = [
    Tool(
        name="GetWordLength",
        func=get_word_length,
        description="文字列の長さを返します。",
    )
]

# %%
agent = create_react_agent(llm, tools, prompt)

# %%
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %%
agent_executor.invoke({"input": "abcの文字数を教えてください"})

# %%
