from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Literal, TypedDict

import dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError, model_validator

dotenv.load_dotenv()


DEFAULT_EVALUATION_CRITERIA: Dict[str, list[str]] = {
    "behavior": [
        "ヒアリング前の適切な導入ができているか",
        "提案後にユーザーの要望へ応じた調整ができているか",
        "旅行と無関係な話題を適切に扱えているか",
    ],
    "quality": [
        "提案内容がユーザーの希望に合致しているか",
        "目的地の説明が十分か",
        "交通手段の提案が妥当か",
        "予算へ配慮できているか",
        "旅行目的に沿った構成か",
        "提案理由が明確か",
        "変更要望に応えられているか",
    ],
}


def _parse_evaluation_criteria_text(text: str) -> Dict[str, list[str]]:
    """Parse YAML text into category -> list of criteria with graceful fallback."""
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None  # type: ignore

    def _fallback_parse(raw: str) -> Dict[str, list[str]]:
        """Legacy bullet parser for backward compatibility."""
        sections: Dict[str, list[str]] = {}
        current: str | None = None
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]") and len(line) > 2:
                current = line[1:-1].strip()
                sections.setdefault(current, [])
                continue
            if line.startswith("-"):
                bullet = line.lstrip("-").strip()
                if not bullet:
                    continue
                target = current or "quality"
                sections.setdefault(target, []).append(bullet)
        return {key: value for key, value in sections.items() if value}

    if yaml is None:
        return _fallback_parse(text)

    try:
        loaded = yaml.safe_load(text) or {}
    except Exception:
        return _fallback_parse(text)

    sections: Dict[str, list[str]] = {}
    if isinstance(loaded, dict):
        for category, items in loaded.items():
            if not isinstance(items, list):
                continue
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            if cleaned:
                sections[str(category)] = cleaned
    elif isinstance(loaded, list):
        # Treat top-level list as quality criteria when no mapping provided.
        cleaned = [str(item).strip() for item in loaded if str(item).strip()]
        if cleaned:
            sections["quality"] = cleaned
    else:
        return _fallback_parse(text)

    return sections or _fallback_parse(text)


def _load_evaluation_criteria() -> Dict[str, list[str]]:
    """Load evaluation criteria from file path in env or fall back to defaults."""
    path = os.environ.get("EVALUATION_CRITERIA_PATH")
    if not path:
        return DEFAULT_EVALUATION_CRITERIA
    try:
        with open(path, "r", encoding="utf-8") as handle:
            parsed_text = handle.read()
    except OSError:
        return DEFAULT_EVALUATION_CRITERIA
    parsed = _parse_evaluation_criteria_text(parsed_text)
    # ensure required categories exist; fall back to defaults if absent
    criteria: Dict[str, list[str]] = {}
    for category, items in parsed.items():
        if items:
            criteria[category] = items
    for category, default_items in DEFAULT_EVALUATION_CRITERIA.items():
        criteria.setdefault(category, default_items)
    return criteria


EVALUATION_CRITERIA = _load_evaluation_criteria()


def _build_evaluation_key_map(criteria: Dict[str, list[str]]) -> Dict[str, Dict[str, str]]:
    key_map: Dict[str, Dict[str, str]] = {}
    for category, items in criteria.items():
        mapping: Dict[str, str] = {}
        for idx, name in enumerate(items, start=1):
            key = f"{category}_{idx}"
            mapping[key] = name
        key_map[category] = mapping
    return key_map


EVALUATION_KEY_MAP = _build_evaluation_key_map(EVALUATION_CRITERIA)
EVALUATION_REQUIRED_KEYS = {
    category: set(mapping.keys()) for category, mapping in EVALUATION_KEY_MAP.items()
}
EVALUATION_SECTION_TITLES = {
    "behavior": "行動",
    "quality": "品質",
}
EVALUATION_SECTION_ORDER = [*EVALUATION_CRITERIA.keys()]


def _format_evaluation_prompt(
    criteria: Dict[str, list[str]],
    key_map: Dict[str, Dict[str, str]],
) -> str:
    lines: list[str] = [
        "あなたは旅行支援アシスタントの評価担当です。",
        "最新の提案を確認し、各評価項目に対して OK または NG を必ず付与してください。",
        "評価カテゴリとキーは次の通りです。",
    ]
    for category in EVALUATION_SECTION_ORDER:
        items = criteria.get(category, [])
        if not items:
            continue
        title = EVALUATION_SECTION_TITLES.get(category, category)
        lines.append(f"- {title} ({category}):")
        for key, name in key_map.get(category, {}).items():
            lines.append(f"  - {key}: {name}")
    lines.append(
        "出力はJSONのみで、次の形式を厳守してください。各キーの値は \"OK\" または \"NG\" を設定します。"
    )
    example_lines: list[str] = ['{"mode":"evaluation",']
    for idx, category in enumerate(EVALUATION_SECTION_ORDER):
        keys = list(key_map.get(category, {}).keys())
        if keys:
            inner = ", ".join(f'"{key}":"OK/NG"' for key in keys)
            line = f' "{category}":{{{inner}}},'
        else:
            line = f' "{category}":{{}},'
        example_lines.append(line)
    example_lines.append(' "comments":"全体のフィードバック"}')
    lines.append("\n".join(example_lines))
    return "\n".join(lines)


def _summarize_section(
    section: str,
    values: Dict[str, str] | None,
    key_map: Dict[str, Dict[str, str]],
) -> str:
    mapping = key_map.get(section, {})
    if not mapping:
        return "-"
    ordered_keys = list(mapping.keys())
    display_items: list[str] = []
    for key in ordered_keys:
        label = mapping[key]
        result = (values or {}).get(key, "?")
        display_items.append(f"{label}:{result}")
    return " / ".join(display_items)

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
            expected_behavior = EVALUATION_REQUIRED_KEYS.get("behavior", set())
            expected_quality = EVALUATION_REQUIRED_KEYS.get("quality", set())
            missing_behavior = expected_behavior - set(behavior.keys())
            missing_quality = expected_quality - set(quality.keys())
            extra_behavior = set(behavior.keys()) - expected_behavior
            extra_quality = set(quality.keys()) - expected_quality
            valid_values = {"OK", "NG"}
            invalid_behavior = [v for v in behavior.values() if v not in valid_values]
            invalid_quality = [v for v in quality.values() if v not in valid_values]
            if (
                missing_behavior
                or missing_quality
                or extra_behavior
                or extra_quality
                or invalid_behavior
                or invalid_quality
                or not (self.comments and self.comments.strip())
            ):
                raise ValueError("mode=evaluation では全ての観点とコメントが必須です。")
            self.message = None
        return self


class ControllerDecision(BaseModel):
    mode: Literal["conversation", "change_request", "off_topic", "evaluation"]
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


class GraphState(TypedDict):
    messages: list[Dict[str, str]]
    tester_state: TesterState
    controller_decision: ControllerDecision | None
    turn_count: int
    turn_limit: int
    evaluation: TesterOutput | None


CONTROLLER_SYS = """
あなたは旅行支援アシスタントのテスターを統括するコントローラです。
与えられた会話ログとテスター内部状態を確認し、次に行うモードを決めてください。
- conversation: ヒアリング段階。旅行に関する通常のユーザーメッセージを返させる。
- off_topic: 旅行と無関係な雑談を1度だけ挟む。state.off_topic_used が true の場合は選ばない。
- change_request: アシスタントが初回プランを出した直後に一度だけ選択し、変更依頼を伝える。
- evaluation: すべてのやり取りが終わり、アシスタントの最終プランを評価するとき。
change_request と off_topic はそれぞれ1回のみ。評価は変更依頼への対応後に行う。
出力はJSONのみで、{"mode": "...","reason":"..."} の形式。
""".strip()


CONVERSATION_PERSONA = """
あなたはゴルフが大好きな30代前半の会社員で、親しみやすい砕けた口調で話す。妻と2人での1泊2日の国内旅行を計画中。自然の中でリフレッシュしたいと考えています。
""".strip()


CONVERSATION_SYS = f"""
あなたは旅行支援アシスタントと会話するユーザーです。常に日本語で自然な1文を返してください。
- {CONVERSATION_PERSONA}
- mode が conversation のときは旅行相談に必要な情報提供や通常のやり取りを行う。
- mode が off_topic のときは旅行と関係ない雑談を短く1文で行う。
- mode が change_request のとき、直前のアシスタント提案に対する具体的な修正要望を1つだけ述べる。
- 要望は同時に最大2つまでに留める。
出力はJSONのみで {{"message":"..."}}。
""".strip()


EVALUATION_SYS = _format_evaluation_prompt(EVALUATION_CRITERIA, EVALUATION_KEY_MAP)


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
    mode: Literal["conversation", "change_request", "off_topic"],
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


def planner_node(state: GraphState) -> Dict[str, Any]:
    if state["turn_count"] >= state["turn_limit"]:
        raise RuntimeError("Turn limit reached before evaluation.")
    assistant_prompt = [{"role": "system", "content": ASSISTANT_SYS}, *state["messages"]]
    debug_log("assistant_prompt", assistant_prompt)
    start = time.perf_counter()
    response = planner_llm.invoke(assistant_prompt)
    latency = (time.perf_counter() - start) * 1000
    content = extract_content(response.content)
    print(f"[Latency] planner_llm: {latency:.2f} ms")
    print(f"[Assistant] {content}")
    new_messages = [*state["messages"], {"role": "assistant", "content": content}]
    return {
        "messages": new_messages,
        "turn_count": state["turn_count"] + 1,
    }


def controller_node(state: GraphState) -> Dict[str, Any]:
    tester_state = replace(state["tester_state"])
    decision = invoke_controller(state["messages"], tester_state)
    print(f"[Controller] mode={decision.mode} reason={decision.reason}")
    return {"controller_decision": decision, "tester_state": tester_state}


def conversation_node(state: GraphState) -> Dict[str, Any]:
    decision = state["controller_decision"]
    if decision is None:
        raise RuntimeError("conversation_node invoked without controller decision.")
    if decision.mode == "evaluation":
        return {"controller_decision": None}

    tester_state = replace(state["tester_state"])
    convo = invoke_conversation_agent(
        state["messages"],
        tester_state,
        decision.mode,
    )
    print(f"[User] {convo.message}")

    if decision.mode == "change_request":
        tester_state.change_request_sent = True
    if decision.mode == "off_topic":
        tester_state.off_topic_used = True

    new_messages = [*state["messages"], {"role": "user", "content": convo.message}]
    return {
        "messages": new_messages,
        "tester_state": tester_state,
        "controller_decision": None,
    }


def evaluation_node(state: GraphState) -> Dict[str, Any]:
    evaluation = invoke_evaluator(state["messages"])
    behavior_summary = _summarize_section("behavior", evaluation.behavior, EVALUATION_KEY_MAP)
    quality_summary = _summarize_section("quality", evaluation.quality, EVALUATION_KEY_MAP)
    print(
        "[[評価結果]]\n"
        f"行動: {behavior_summary}\n"
        f"品質: {quality_summary}\n"
        f"コメント: {evaluation.comments}"
    )
    return {
        "evaluation": evaluation,
        "controller_decision": None,
    }


def _route_from_controller(state: GraphState) -> str:
    decision = state["controller_decision"]
    if decision is None:
        raise RuntimeError("controller decision missing when routing.")
    return decision.mode


def build_multiagent_graph() -> Any:
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("controller", controller_node)
    graph.add_node("conversation", conversation_node)
    graph.add_node("evaluation", evaluation_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "controller")
    graph.add_conditional_edges(
        "controller",
        _route_from_controller,
        {
            "conversation": "conversation",
            "change_request": "conversation",
            "off_topic": "conversation",
            "evaluation": "evaluation",
        },
    )
    graph.add_edge("conversation", "planner")
    graph.add_edge("evaluation", END)
    return graph.compile()


MULTIAGENT_GRAPH = build_multiagent_graph()


def run_simulation(turn_limit: int = 15) -> TesterOutput | None:
    initial_state: GraphState = {
        "messages": [{"role": "user", "content": "こんにちは。旅行の相談をしたいです。"}],
        "tester_state": TesterState(),
        "controller_decision": None,
        "turn_count": 0,
        "turn_limit": turn_limit,
        "evaluation": None,
    }
    final_state = MULTIAGENT_GRAPH.invoke(
        initial_state,
        config={"recursion_limit": turn_limit * 2 + 4},
    )
    return final_state.get("evaluation")


def save_multiagent_graph_png(path: str = "multiagent_graph.png") -> None:
    """Render the compiled LangGraph to a PNG file (requires Graphviz)."""
    MULTIAGENT_GRAPH.get_graph().draw_png(path)


def save_multiagent_graph_mermaid(path: str = "multiagent_graph.mmd") -> None:
    """Export the graph as Mermaid text for tools that accept Mermaid diagrams."""
    content = MULTIAGENT_GRAPH.get_graph().draw_mermaid()
    with open(path, "w", encoding="utf-8") as fp:
        fp.write(content)


if __name__ == "__main__":
    run_simulation()
    save_multiagent_graph_mermaid()
