# AI対AIの会話をシミュレーションする
# デバッグモードとpydanticモデルによる出力検証を追加

# %%
from langchain_openai import ChatOpenAI
import dotenv
import json
import os
import openai
import time

dotenv.load_dotenv()

# OpenAI APIキーの設定
openai.api_key = os.environ.get("OPENAI_API_KEY")


# %%
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Any, Dict, Literal, Optional


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

    @model_validator(mode="after")
    def _check_mode_fields(self):
        if self.mode == "user_message":
            if not (self.message and self.message.strip()):
                raise ValueError("mode=user_message のとき message は必須です。")
            # 他フィールドは None に揃える
            self.behavior = None
            self.quality = None
            self.comments = None
        elif self.mode == "evaluation":
            behavior = self.behavior or {}
            quality = self.quality or {}
            required_behavior = {"pre_hearing", "post_hearing_plan", "reject_irrelevant"}
            required_quality = {"match", "destination", "budget", "transport", "purpose", "explain_match"}
            missing_behavior = required_behavior - set(behavior.keys())
            missing_quality = required_quality - set(quality.keys())
            if missing_behavior or missing_quality or not (self.comments and self.comments.strip()):
                raise ValueError(
                    f"mode=evaluation のとき behavior/quality/comments は必須です。"
                    f" missing_behavior={missing_behavior}, missing_quality={missing_quality}"
                )
            self.message = None
        return self

# %%
tester_llm = ChatOpenAI(
    model_name="gpt-5",
    openai_api_key=openai.api_key,
    temperature=0.0,
    max_tokens=4096,
)
planner_llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=openai.api_key,
    temperature=0.0,
    max_tokens=512,
)

# DEBUGフラグは環境変数で制御（未設定なら無効）
DEBUG = os.environ.get("TESTER_DEBUG", "0").lower() not in {"0", "false", ""}


def debug_log(label: str, payload) -> None:
    if not DEBUG:
        return
    print(f"[DEBUG] {label}: {payload}")


def extract_content(content: Any) -> str:
    """LangChainのmessage.contentがリストで返るケースを吸収する。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                texts.append(chunk["text"])
        return "".join(texts)
    return str(content)


def invoke_tester(conversation_messages):
    prompt = [{"role": "system", "content": TESTER_SYS}, *conversation_messages]
    debug_log("tester_prompt", prompt)
    start = time.perf_counter()
    response = tester_llm.invoke(prompt)
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"[Latency] tester_llm: {latency_ms:.2f} ms")
    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    debug_log("tester_response_meta", additional_kwargs)
    debug_log("tester_response_object", response)
    raw_content = extract_content(response.content)
    if not raw_content.strip():
        if "tool_calls" in additional_kwargs:
            tool_calls = additional_kwargs.get("tool_calls") or []
            if tool_calls:
                raw_content = tool_calls[0].get("function", {}).get("arguments", "") or raw_content
        if not raw_content.strip() and "function_call" in additional_kwargs:
            raw_content = additional_kwargs["function_call"].get("arguments", "") or raw_content
    debug_log("tester_raw_response", raw_content)
    try:
        parsed = TesterOutput.model_validate_json(raw_content)
    except ValidationError as exc:
        debug_log("tester_validation_error", exc)
        raise RuntimeError(f"TesterOutput parse failed: {raw_content}") from exc
    debug_log("tester_parsed", parsed)
    return parsed
TESTER_SYS = """
あなたは旅行支援アシスタントbotのテスターです。

指示
- アシスタントとの会話ログを入力します。あなたは**ユーザー役**として旅行支援アシスタントにメッセージを1つ返すか、**評価者役**としてアシスタントの直前の出力に対する評価を返します。
- モード切り替えルールを参照し、会話モードと評価モードを切り替えて指示にしたがった応答をしてください。
- 常にJSONのみを出力してください（前後の説明・コードブロック禁止）。
- 会話モードでは behavior/quality/comments を必ず null にしてください。評価モードでは message を必ず null にしてください。

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
debug_log("conversation_start", messages)
for i in range(15):
    if i % 2 != 0:
        # TesterAgentのターン
        tester_output = invoke_tester(messages)
        if tester_output.mode == "user_message":
            print(f"[User] {tester_output.message}")
            messages.append({"role": "user", "content": tester_output.message or ""})
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
        assistant_prompt = [{"role": "system", "content": system}, *messages]
        debug_log("assistant_prompt", assistant_prompt)
        assistant_start = time.perf_counter()
        assistant_output = planner_llm.invoke(assistant_prompt)
        assistant_latency_ms = (time.perf_counter() - assistant_start) * 1000
        print(f"[Latency] planner_llm: {assistant_latency_ms:.2f} ms")
        assistant_content = extract_content(assistant_output.content)
        debug_log("assistant_raw_response", assistant_content)
        print(f"[Assistant] {assistant_content}")
        messages.append({"role": "assistant", "content": assistant_content})


# %%
