# %%
import langchain
from dotenv import load_dotenv

load_dotenv()
# %%
from langchain_tavily import TavilySearch

search = TavilySearch(max_results=5)
search_results = search.invoke("東京の天気は？")
print(search_results)
from datetime import date
from langchain.tools import Tool


def get_today_date(_: str = "") -> str:
    """今日の日付を ISO 形式 (YYYY-MM-DD) で返します。引数は無視されます。"""
    return date.today().isoformat()


# ツール一覧（検索ツールに加えて今日の日付を返すツールを追加）
today_tool = Tool(name="GetTodayDate", func=get_today_date, description="今日の日付を ISO 形式で返します。")
tools = [search, today_tool]
# %%
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.0)

# # %%
# query = "こんにちは!"
# response = model.invoke([{"role": "user", "content": query}])
# response.text()
# # %%
# # ツールのバインド
# model_with_tools = model.bind_tools(tools)
# query = "こんにちは!"
# response = model_with_tools.invoke([{"role": "user", "content": query}])

# print(f"Message content: {response.text()}\n")
# print(f"Tool calls: {response.tool_calls}")
# # %%
# query = "東京の天気を調べてください"
# response = model_with_tools.invoke([{"role": "user", "content": query}])

# # ツールが実行されるがメッセージは表示されない
# print(f"Message content: {response.text()}\n")
# print(f"Tool calls: {response.tool_calls}")
# %%[markdown]
#### ここからエージェントを作っていく

# %%
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "aaa123"}}

system_prompt = """
あなたは日本語で応答するアシスタントです。ユーザーの問い合わせで必要な場合は下のツールを使って情報を取得してください。ツールは信頼できる外部検索/日時ツールのみを使えます。ツールを使う基準：
- 正確な時刻・日付が必要な場合は必ず GetTodayDate を呼ぶ。
- 事実確認や最新の情報が必要な場合は TavilySearch を呼ぶ（地名を含めて検索する）。
応答ルール：
1. ユーザーへの最終応答は日本語で簡潔に1段落で回答する。
2. ツールを呼び出したら、ツール結果を必ず要約してからユーザーに返す。
3. ツールの結果は必要に応じて日本語に直す（もし英語なら）。

例1:
User: "今日の日付は？"
Tool: GetTodayDate -> "2025-09-28"
Assistant: "今日の日付は 2025-09-28 です。"

例2:
User: "9月28日の新宿区の天気を教えて"
Tool: TavilySearch("新宿区 天気 2025年9月28日") -> <英語結果のテキスト>
Assistant: "<日本語要約>（出典: TavilySearch）"

出力形式（例）：
- まずツール呼び出しがあれば「ツール実行: <ツール名>」を内部で実行してから、最終的に「回答: <日本語の本文>」だけをユーザーに返してください。
"""
system_message = {"role": "system", "content": system_prompt}
thread_id="test1234"

# %%
def call_agent_with_system(user_content, thread_id="aaa123"):
    # システムメッセージとユーザーメッセージを組み合わせてエージェントを呼び出す
    system_message = {"role":"system","content": system_prompt}
    user_message = {"role":"user","content": user_content}
    cfg = {"configurable": {"thread_id": thread_id}}
    return agent_executor.stream({"messages":[system_message, user_message]}, config=cfg, stream_mode="values")

# %%
for step in call_agent_with_system("こんにちは!今日は近所の公園に野球をしにいく予定なんだよね", thread_id="test123"):
    step["messages"][-1].pretty_print()
# %%
for step in call_agent_with_system("今日の新宿の天気を検索して教えてくれる？", thread_id="test123"):
    step["messages"][-1].pretty_print()

# %%
