"""Phase 2: ISL gloss sequence -> natural English, as a small LangGraph graph (reverse of gloss_graph)."""
import logging
import re
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.gloss_graph import _llm

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You translate Indian Sign Language (ISL) gloss into natural, fluent English.

The input is a sequence of UPPERCASE gloss words recognised from signing, one per sign, in signing order.
ISL grammar differs from English:
- Word order is time first, then topic, then comment, roughly Subject-Object-Verb.
- Question words (WHAT, WHERE, WHO, WHEN, WHY, HOW) come at the END.
- NOT comes after the verb it negates; FINISH after a verb marks completed/past action.
- There are no articles, no forms of "to be", and no tense endings - add them as English needs.
- Pronouns: I/ME, YOU, HE, SHE, WE, THEY, MY, YOUR. Hyphenated glosses are one sign (GOOD-MORNING).
- Recognition can be imperfect: if a word clearly does not fit, you may drop it, but never invent content.

Output ONLY the English sentence (one sentence, or two short ones if the signs clearly form two). No quotes, no notes."""

FEW_SHOT: list[tuple[str, str]] = [
    ("YOUR NAME WHAT", "What is your name?"),
    ("TOMORROW I SCHOOL GO", "I am going to school tomorrow."),
    ("I UNDERSTAND NOT", "I don't understand."),
    ("YOU FOOD EAT FINISH", "Have you eaten?"),
    ("HOSPITAL WHERE", "Where is the hospital?"),
    ("GOOD-MORNING HOW YOU", "Good morning, how are you?"),
    ("MY FATHER DOCTOR", "My father is a doctor."),
    ("YESTERDAY WE MARKET GO FINISH", "We went to the market yesterday."),
    ("I WATER WANT", "I want some water."),
]


class EnglishState(TypedDict, total=False):
    gloss: str
    english: str


def _messages(gloss: str) -> list[BaseMessage]:
    msgs: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT)]
    for g, e in FEW_SHOT:
        msgs += [HumanMessage(g), AIMessage(e)]
    msgs.append(HumanMessage(gloss))
    return msgs


def translate_node(state: EnglishState) -> EnglishState:
    out = str(_llm().invoke(_messages(state["gloss"])).content).strip()
    out = re.sub(r'^(english\s*:\s*)', "", out, flags=re.I).strip().strip('"').strip()
    return {"english": out.splitlines()[0] if out else ""}


_graph = None


def gloss_to_english(glosses: list[str]) -> str:
    global _graph
    gloss = " ".join(g.strip().upper() for g in glosses if g.strip())
    if not gloss:
        return ""
    if _graph is None:
        g = StateGraph(EnglishState)
        g.add_node("translate", translate_node)
        g.add_edge(START, "translate")
        g.add_edge("translate", END)
        _graph = g.compile()
    english = _graph.invoke({"gloss": gloss})["english"]
    log.info("English: %r -> %r", gloss, english)
    return english
