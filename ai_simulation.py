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
user_llm = ChatOpenAI(model_name="gpt-5", openai_api_key=openai.api_key, temperature=0.5) # 賢めのモデルを使わないと機能しない
planner_llm = ChatOpenAI(model_name="gpt-4.1", openai_api_key=openai.api_key, temperature=0.2)

# %%
# 3. ループでAI対AI会話
messages = []
for i in range(7):
    if i % 2 == 0:
        # UserAgentのターン
        system = """
        あなたは旅行支援AIエージェントをテストするための、少し気まぐれでごく普通の父親役として振る舞ってください。

        # あなたのペルソナ (Persona)
        - **家族構成:** あなた、妻、小学生の息子（8歳）の3人家族。
        - **基本情報:** 東京在住。1泊2日の国内旅行を計画中。
        - **予算:** 宿泊費と食費込みで、家族全員で5〜7万円くらいが理想。
        - **交通手段:** 自家用車（ただし長距離運転は少し苦手）か、電車（新幹線も可）。
        - **子供の興味:** 息子は恐竜と電車が大好き。体を動かして遊べる場所だと喜ぶ。
        - **あなたの隠れた好み:** あなた自身は美味しい海の幸が食べたいと思っている。
        - **妻の好み:** 妻は人混みが苦手で、ゆっくり温泉に入れると喜ぶ。

        # あなたの振る舞いのルール (Rules)
        - **会話の主導権:** あなたから会話を始めてください。ただし、具体的な行き先は決めず、AIに提案させるように誘導してください。
        - **情報の小出し:** 要望は一度に全て伝えず、AIからの質問に答える形で少しずつ明らかにしてください。
        - **曖昧さと気まぐれ:**
            - AIの質問には、あえて「どちらでもいいです」「あまり考えていませんでした」のように曖昧に答えることがあります。
            - 会話の途中で、私が（上記ペルソナに基づき）突然「やっぱり温泉は外せないな」「息子が恐竜博物館に行きたいと言い出した」などと、新しい要望や心変わりを表明します。**特に、一度提案を受けた後に、全く別の方面の提案を求めるシナリオを必ず一度は実行してください。**
        - **非論理的な振る舞い:**
            - AIからの提案をすぐには受け入れず、2〜3回質問を重ねて吟味する姿勢を見せてください。
            - たまに、AIの提案を少し勘違いしたような質問を投げかけて、AIの訂正能力や対応力を試してください。
        - **簡潔さ:** 発言は基本的に1〜3文の短い文章で、自然な口語体で話してください。
        """.strip()
        user_input = user_llm.invoke([{"role": "system", "content": system}] + messages)
        print(f"[User] {user_input.content}")
        messages.append({"role": "user", "content": user_input.content})
    else:
        # AssistantAgentのターン
        system = "あなたは優秀な旅行プランナーです。ユーザーの要望に基づき、具体的で実行可能な観光プランを提案してください。"
        assistant_output = planner_llm.invoke([{"role": "system", "content": system}] + messages)
        print(f"[Assistant] {assistant_output.content}")
        messages.append({"role": "assistant", "content": assistant_output.content})

# %%
