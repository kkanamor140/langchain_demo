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
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Optional


class TesterOutput(BaseModel):
    mode: Literal["user_message", "evaluation"]
    # user_message 用
    message: Optional[str] = Field(
        default=None, description="ユーザーとして出す1文の日本語メッセージ"
    )
    # evaluation 用
    behavior: Optional[Dict[str, str]] = None  # OK/NG
    quality: Optional[Dict[str, str]] = None   # OK/NG
    comments: Optional[str] = None

    # @model_validator(mode="after")
    # def _check_mode_fields(self):
    #     if self.mode == "user_message":
    #         if not self.message:
    #             raise ValueError("mode=user_message のとき message は必須です。")
    #         # 他方は None に寄せる
    #         self.behavior = None
    #         self.quality = None
    #         self.comments = None
    #     elif self.mode == "evaluation":
    #         if not (self.behavior and self.quality and self.comments):
    #             raise ValueError("mode=evaluation のとき behavior/quality/comments は必須です。")
    #         self.message = None
    #     return self

# %%
tester_llm: TesterOutput = (
    ChatOpenAI(
        model_name="gpt-5", openai_api_key=openai.api_key, temperature=0.5, max_tokens=4096)
    .with_structured_output(TesterOutput, strict=True)
)  # 賢めのモデルを使わないと機能しない
planner_llm = ChatOpenAI(model_name="gpt-4.1", openai_api_key=openai.api_key, temperature=0.2, max_tokens=4096)

TESTER_SYS = """
あなたは旅行支援アシスタントbotのテスターです。

指示
- アシスタントとの会話ログを入力します。あなたは**ユーザー役**として旅行支援アシスタントにメッセージを1つ返すか、**評価者役**としてアシスタントの直前の出力に対する評価を返します。
- モードリ切り替えルールに従い会話モードと評価モードを切り替えて指示にしたがった応答をしてください。
- 常にJSONのみを出力してください（前後の説明・コードブロック禁止）。

モード切替ルール:
- 直前のアシスタント出力がプランの提案であれば「評価モード」で次のJSONを返す:
{"mode":"evaluation",
 "behavior":{"pre_hearing":"OK/NG","post_hearing_plan":"OK/NG","reject_irrelevant":"OK/NG"},
 "quality":{"match":"OK/NG","destination":"OK/NG","budget":"OK/NG","transport":"OK/NG","purpose":"OK/NG","explain_match":"OK/NG"},
 "comments":"..."}
- それ以外は「会話モード」でアシスタントに送信するメッセージを含む次のJSONを**1つのみ**返却し、アシスタントの応答が入力されるまで待機してください:
{"mode":"user_message","message":"..."}

会話モードの制約:
- 出力は必ず日本語
- messageは自然な会話になるように心がける
- 基本は受け身で、質問には簡潔に回答する
- 会話中に1回だけ旅行と無関係の話題を振る（既に実施済みなら二度としない）。
- 要望を同時に伝える場合は最大2つまで。

評価モードの観点:
- 具体的提案前にヒアリングがあったか（pre_hearing）
- ヒアリング後にプラン提示したか（post_hearing_plan）
- 無関係話題を拒否できたか（reject_irrelevant）
- ヒアリング結果に沿っているか（match）
- 目的地/予算/交通/目的を考慮できているか
- 要望に合致していることの説明（explain_match）
""".strip()

# %%
# 3. ループでAI対AI会話
messages = [{"role": "user", "content": "こんにちは。旅行の相談をしたいです。"}]
for i in range(15):
    if i % 2 != 0:
        # TesterAgentのターン
        # print([{"role": "system", "content": TESTER_SYS}, *messages])
        tester_output = tester_llm.invoke([{"role": "system", "content": TESTER_SYS}, *messages])
        if tester_output.mode == "user_message":
            print(f"[User] {tester_output.message}")
            messages.append({"role": "user", "content": f"[User] {tester_output.message}"})
        else:
            print(f"[[評価結果]]\n行動: {tester_output.behavior}\n品質: {tester_output.quality}\nコメント: {tester_output.comments}")
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
        assistant_output = planner_llm.invoke([{"role": "system", "content": system}, *messages])
        print(f"[Assistant] {assistant_output.content}")
        messages.append({"role": "assistant", "content": f"[Assistant] {assistant_output.content}"})

# %%
tester_output

# %%
