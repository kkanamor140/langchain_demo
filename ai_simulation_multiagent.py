# Prototype: Separate tester controller, conversation agent, and evaluation agent
# This is a prototype to explore splitting the tester logic into dedicated roles.

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal

import dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

dotenv.load_dotenv()

# Shared utility: extract LangChain content regardless of structure
def extract_content(content: Any) -> str:
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


def debug_log(label: str, payload: Any) -> None:
    if os.environ.get("TESTER_DEBUG", "0").lower() in {"0", "false", ""}:
        return
    print(f"[DEBUG] {label}: {payload}")


class TesterOutput(BaseModel):
    mode: Literal["user_message", "evaluation"]
    message: str | None = Field(
        default=None, description="ユーザーとして出す1文の日本語メッセージ"
    )
    behavior: Dict[str, str] | None = None
    quality: Dict[str, str] | None = None
    comments: str | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self):
        if self.mode == "user_message":
            if not (self.message and self.message.strip()):
                raise ValueError("mode=user_message のとき message は必須です。")
            self.behavior = None
            self.quality = None
            self.comments = None
        elif self.mode == "evaluation":
            behavior = self.behavior or {}
            quality = self.quality or {}
            required_behavior = {"pre_hearing", "post_hearing_plan", "reject_irrelevant"}
            required_quality = {
                "match",
                "destination",
                "budget",
                "transport",
                "purpose",
                "explain_match",
                "change_request",
            }
            missing_behavior = required_behavior - set(behavior.keys())
            missing_quality = required_quality - set(quality.keys())
            if missing_behavior or missing_quality or not (self.comments and self.comments.strip()):
                raise ValueError(
                    "mode=evaluation では全ての観点とコメントが必須です。"
                )
            self.message = None
        return self


class ControllerDecision(BaseModel):
    mode: Literal["conversation", "change_request", "evaluation"]
    reason: str


class ConversationOutput(BaseModel):
    message: str

    @model_validator(mode="after")
    def _validate_message(self):
        if not self.message.strip():
            raise ValueError("message が空です。")
        return self


tester_controller_llm = ChatOpenAI(
    model_name="gpt-5-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
    max_tokens=4096,
)
tester_conversation_llm = ChatOpenAI(
    model_name="gpt-5-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
    max_tokens=4096,
)
tester_evaluator_llm = ChatOpenAI(
    model_name="gpt-5-mini",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
    max_tokens=4096,
)
planner_llm = ChatOpenAI(
    model_name="gpt-4.1",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=0.0,
    max_tokens=4096,
)


@dataclass
class TesterState:
    change_request_sent: bool = False
    off_topic_used: bool = False


CONTROLLER_SYS = """
あなたは旅行支援アシスタントのテスターを統括するコントローラです。
与えられた会話ログとテスター内部状態を確認し、次に行うモードを決めてください。
- conversation: ヒアリング段階または雑談時。通常のユーザーメッセージを返させる。
- change_request: アシスタントが初回プランを出した直後に一度だけ選択し、変更依頼を伝える。
- evaluation: すべてのやり取りが終わり、アシスタントの最終プランを評価するとき。
change_request は1回のみ。評価は変更依頼への対応後に行う。
出力はJSONのみで、{"mode": "...","reason":"..."} の形式。
""".strip()


CONVERSATION_SYS = """
あなたは旅行支援アシスタントと会話するユーザーです。常に日本語で自然な1文を返してください。
- mode が change_request のとき、直前のアシスタント提案に対する具体的な修正要望を1つだけ述べる。
- mode が conversation のときは通常のヒアリング回答や雑談を行う。
- travel unrelated な話題は会話中に1度だけ行い、状態 off_topic_used が true なら再び触れない。
- 要望は同時に最大2つまでに留める。
出力はJSONのみで {"message":"..."}。
""".strip()


EVALUATION_SYS = """
あなたは旅行支援アシスタントの評価担当です。
アシスタントの最新プランを評価し、必ず次のJSONを返してください。
{"mode":"evaluation",
 "behavior":{"pre_hearing":"OK/NG","post_hearing_plan":"OK/NG","reject_irrelevant":"OK/NG"},
 "quality":{"match":"OK/NG","destination":"OK/NG","budget":"OK/NG","transport":"OK/NG","purpose":"OK/NG","explain_match":"OK/NG","change_request":"OK/NG"},
 "comments":"全体のフィードバック"}
コメントは日本語で簡潔に。
""".strip()


def invoke_controller(messages: list[Dict[str, str]], state: TesterState) -> ControllerDecision:
    payload = json.dumps(
        {
            "state": state.__dict__,
            "history": messages,
        },
        ensure_ascii=False,
    )
    prompt = [
        {"role": "system", "content": CONTROLLER_SYS},
        {"role": "user", "content": payload},
    ]
    debug_log("controller_prompt", prompt)
    response = tester_controller_llm.invoke(prompt)
    raw = extract_content(response.content)
    debug_log("controller_raw", raw)
    try:
        parsed = ControllerDecision.model_validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"ControllerDecision parse failed: {raw}") from exc
    return parsed


def invoke_conversation_agent(
    messages: list[Dict[str, str]],
    state: TesterState,
    mode: Literal["conversation", "change_request"],
) -> ConversationOutput:
    payload = json.dumps(
        {
            "mode": mode,
            "state": state.__dict__,
            "history": messages,
        },
        ensure_ascii=False,
    )
    prompt = [
        {"role": "system", "content": CONVERSATION_SYS},
        {"role": "user", "content": payload},
    ]
    debug_log("conversation_prompt", prompt)
    response = tester_conversation_llm.invoke(prompt)
    raw = extract_content(response.content)
    debug_log("conversation_raw", raw)
    try:
        parsed = ConversationOutput.model_validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"ConversationOutput parse failed: {raw}") from exc
    return parsed


def invoke_evaluator(messages: list[Dict[str, str]]) -> TesterOutput:
    payload = json.dumps({"history": messages}, ensure_ascii=False)
    prompt = [
        {"role": "system", "content": EVALUATION_SYS},
        {"role": "user", "content": payload},
    ]
    debug_log("evaluation_prompt", prompt)
    response = tester_evaluator_llm.invoke(prompt)
    raw = extract_content(response.content)
    debug_log("evaluation_raw", raw)
    try:
        parsed = TesterOutput.model_validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"TesterOutput parse failed: {raw}") from exc
    return parsed


ASSISTANT_SYS = """
あなたは旅行支援エージェントです。
会話ログの最後のユーザーメッセージを参照し、以下に従って応答してください。
- ヒアリング項目（目的地・交通手段・旅の目的）が揃っていなければ1問だけ質問する。
- すべて揃っていれば最適なプランを1つ提案する。
- 旅行と無関係な話題には丁寧に応答を断る。
出力は日本語で自然な文章にする。
""".strip()


def run_simulation(turn_limit: int = 15) -> None:
    messages: list[Dict[str, str]] = [
        {"role": "user", "content": "こんにちは。旅行の相談をしたいです。"}
    ]
    state = TesterState()
    for turn in range(turn_limit):
        if turn % 2 == 0:
            # Assistant turn
            assistant_prompt = [{"role": "system", "content": ASSISTANT_SYS}, *messages]
            debug_log("assistant_prompt", assistant_prompt)
            start = time.perf_counter()
            response = planner_llm.invoke(assistant_prompt)
            latency = (time.perf_counter() - start) * 1000
            print(f"[Latency] planner_llm: {latency:.2f} ms")
            content = extract_content(response.content)
            print(f"[Assistant] {content}")
            messages.append({"role": "assistant", "content": content})
        else:
            # Tester controller decides next action
            decision = invoke_controller(messages, state)
            print(f"[Controller] mode={decision.mode} reason={decision.reason}")
            if decision.mode == "evaluation":
                evaluation = invoke_evaluator(messages)
                print(f"[[評価結果]]\n行動: {evaluation.behavior}\n品質: {evaluation.quality}\nコメント: {evaluation.comments}")
                break
            convo = invoke_conversation_agent(messages, state, decision.mode)
            print(f"[User] {convo.message}")
            messages.append({"role": "user", "content": convo.message})
            if decision.mode == "change_request":
                state.change_request_sent = True
            if not state.off_topic_used and "旅行" not in convo.message:
                # 粗い判定だが、旅行に触れていないメッセージを雑談として扱いフラグを立てる
                state.off_topic_used = True

if __name__ == "__main__":
    run_simulation()

