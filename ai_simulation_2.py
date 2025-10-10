# AI対AIの会話をシミュレーションする

# %%
from langchain_openai import ChatOpenAI
import dotenv
import os
import openai

dotenv.load_dotenv()

# OpenAI APIキーの設定
openai.api_key = os.environ.get("OPENAI_API_KEY")

# %%
tester_llm = ChatOpenAI(model_name="gpt-5", openai_api_key=openai.api_key, temperature=0.5, max_completion_tokens=4096) # 賢めのモデルを使わないと機能しない
planner_llm = ChatOpenAI(model_name="gpt-4.1", openai_api_key=openai.api_key, temperature=0.2, max_completion_tokens=4096)

# %%
# 3. ループでAI対AI会話
messages = []
for i in range(15):
    if i % 2 != 0:
        # TesterAgentのターン
        system = """
        あなたは旅行支援エージェントをテスターです。
        旅行支援エージェントはあなたの旅先をヒアリングし、最適な旅行プランを提案します。
        会話ログを入力するので以下のルールに従い会話モードと評価モードを切り替えて指示にしたがった応答をしてください。

        # ルール (Rules)
        - **会話モード:** 会話ログの最後で旅行プランが提案されていない場合、ユーザーとして旅行支援エージェントに送信するメッセージを1つ出力してください。
        - **評価モード:** 会話ログの最後で旅行プランが提案されている場合、提案されたプランに対する評価を出力してください。評価結果は`[[評価結果]]`という書き出しで始めてください。

        # 会話モードにおけるルール (Conversation Mode Rules)
        - 出力は必ず日本語で行ってください
        - エージェントの性能を引き出すために、基本は受け身で、質問には簡潔に答えるようにしてください
        - エージェントの頑健性をテストするため、エージェントの質問を無視して旅行と関係がない話題をふるメッセージを1つ混ぜてください。ただし、1度で十分なので繰り返しはしないでください
        - 一度に1つのメッセージを出力してください
        - メッセージは1つの文で完結させてください
        - エージェントに要望を伝える場合は同時に伝える要望は最大2つまでにしてください
        - アシスタントのような説明や提案は絶対にしないでください。
        
        # 評価モードにおける品質チェック項目 (Quality Check Items)
        - エージェントの振る舞い
            - 具体的な提案を行う前に要望のヒアリングを行っているか (OK / NG)
            - ヒアリングのあとに旅行プランを提案しているか (OK / NG)
            - 旅行に関係のない質問に対しては回答できないことを伝えているか (OK / NG)
        - 旅行プランの品質
            - ヒアリング結果に即した提案を行っているか (OK / NG)
            - 以下の項目が考慮できているか
                - 目的地 (OK / NG)
                - 予算 (OK / NG)
                - 交通手段 (OK / NG)
                - 旅の目的 (OK / NG)
            - 提案内容が要望に合致していることをわかりやすく説明しているか (OK / NG)
        """.strip()
        tester_input = tester_llm.invoke([{"role": "system", "content": system}] + messages)
        print(f"[User] {tester_input.content}")
        messages.append({"role": "user", "content": tester_input.content})
        if tester_input.content.startswith("[[評価結果]]"):
            print("評価終了")
            break
    else:
        # AssistantAgentのターン
        system = """
        あなたは旅行支援エージェントです。

        会話ログを与えるのでユーザーからの最後のメッセージを参照し、以下のルールにしたがった応答をしてください。
        
        ルール
        - ヒアリング項目がすべて確認できていない場合、追加のヒアリングのための質問を1つだけ出力してください。一度に多くは質問せず、ユーザーが頭を悩ませずに答えられるようにしてください
        - ヒアリング項目がすべて確認できている場合、最適な旅行プランを1つだけ提案してください。
        - ユーザーのメッセージが旅行に関係のない話題の場合、旅行に関連しないことには回答できないことを伝えてください
        
        ヒアリング項目
        - 目的地(最優先)
        - 交通手段
        - 旅の目的
        """
        assistant_output = planner_llm.invoke([{"role": "system", "content": system}] + messages)
        print(f"[Assistant] {assistant_output.content}")
        messages.append({"role": "assistant", "content": assistant_output.content})

# %%
