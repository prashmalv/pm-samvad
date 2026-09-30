"""English -> ISL gloss, as a small LangGraph graph around one AzureChatOpenAI node.

Graph:  START -> translate (LLM) -> normalize (deterministic clean-up) -> END
"""
import logging
import re
from functools import lru_cache
from typing import TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

from app import topics
from app.config import azure_settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You convert English sentences into Indian Sign Language (ISL) gloss.

ISL GLOSS RULES
1. Output ONLY the gloss: UPPERCASE words separated by single spaces. No punctuation, no quotes, no explanation.
2. Drop articles (a, an, the) and copulas/auxiliaries (is, am, are, was, were, be, been, do, does, did, will, shall, have/has when auxiliary).
3. Word order: TIME first, then TOPIC, then COMMENT. Within a clause use Subject-Object-Verb (SOV).
4. Question words (WHAT, WHERE, WHO, WHEN, WHY, HOW) go at the END of the sentence.
5. Negation (NOT) comes AFTER the verb or adjective it negates.
6. No tense inflection: use the base form of verbs (went -> GO, eating -> EAT). Mark past with a time word (YESTERDAY) or FINISH after the verb; mark future with a time word (TOMORROW).
7. Pronouns: I (for I/me), YOU, HE, SHE, WE, THEY, MY, YOUR. Plurals use the singular gloss.
8. Multi-word signs are joined by a hyphen, e.g. THANK-YOU.
9. Proper names and words with no sign stay as a single uppercase English word (e.g. RAHUL, DELHI).
10. Say each participant once: do not repeat a pronoun that is already the topic of the sentence.
11. Keep the meaning. Drop filler words (just, really, um, so, well) and politeness padding except PLEASE / THANK-YOU / SORRY.

CONVERSATION TOPIC: {topic}
{context}
Read ambiguous words the way this setting means them.

PREFERRED VOCABULARY
These glosses have recorded video clips. When a word in the sentence has the same or a very close meaning as one of these, use the vocabulary gloss (e.g. "glad"/"pleased" -> HAPPY, "house" -> HOME, "thanks" -> THANK-YOU, "hi" -> HELLO). Never change the meaning just to fit the vocabulary; if nothing fits, use the plain uppercase English word.
Signs for this topic: {topic_vocabulary}
Other signs: {vocabulary}"""

# Few-shot pairs chosen to cover each rule: wh-question, time-first, negation,
# past via FINISH, copula drop, proper name, synonym-to-vocabulary, politeness.
FEW_SHOT: list[tuple[str, str]] = [
    ("What is your name?", "YOUR NAME WHAT"),
    ("My name is Rahul.", "MY NAME RAHUL"),
    ("I am going to school tomorrow.", "TOMORROW I SCHOOL GO"),
    ("Where is the hospital?", "HOSPITAL WHERE"),
    ("I don't understand.", "I UNDERSTAND NOT"),
    ("Have you eaten your food?", "YOU FOOD EAT FINISH"),
    ("The doctor came to our house yesterday.", "YESTERDAY DOCTOR OUR HOME COME"),
    ("I am so glad to meet you!", "I YOU MEET HAPPY"),
    ("Could you please help me?", "PLEASE YOU I HELP"),
    ("Thanks, I don't want any water.", "THANK-YOU I WATER WANT NOT"),
    ("Please help me, I need water.", "I WATER WANT HELP PLEASE"),
    ("Why are you sad today?", "TODAY YOU SAD WHY"),
]

# Safety net in case the model leaks English function words.
_DROP = {"A", "AN", "THE", "IS", "AM", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING", "TO", "OF"}


class GlossState(TypedDict, total=False):
    english: str
    vocabulary: list[str]
    topic: str
    raw_output: str
    gloss: str
    tokens: list[str]


@lru_cache(maxsize=1)
def _llm() -> AzureChatOpenAI:
    s = azure_settings()
    return AzureChatOpenAI(
        azure_endpoint=s["AZURE_OPENAI_ENDPOINT"],
        api_key=s["AZURE_OPENAI_API_KEY"],
        api_version=s["AZURE_OPENAI_API_VERSION"],
        azure_deployment=s["AZURE_OPENAI_DEPLOYMENT_NAME"],
        temperature=0,  # deterministic gloss for the same sentence (gpt-4.1 family supports this)
        max_tokens=120,
        timeout=30,
        max_retries=2,
    )


# Everyday glosses always offered to the LLM (when they have clips), even for a large library.
CORE_VOCAB = set(
    "HELLO THANK-YOU SORRY YES NOT I YOU YOUR HE SHE WE THEY NAME WHAT WHERE WHY HOW WHO WHEN "
    "GOOD HAPPY SAD SICK LOVE HELP WANT GO COME EAT DRINK SLEEP WORK LEARN SEE KNOW UNDERSTAND "
    "FINISH FRIEND FATHER MOTHER BROTHER DOCTOR WATER FOOD BOOK HOME TODAY DAY NIGHT MORNING NEXT BEFORE".split()
)
MAX_FULL_VOCAB = 300


def _relevant_vocab(english: str, vocabulary: list[str]) -> list[str]:
    """For big libraries, send only glosses related to the sentence (shared 4-letter word stems) plus core words."""
    if len(vocabulary) <= MAX_FULL_VOCAB:
        return vocabulary
    stems = {w[:4] for w in re.findall(r"[a-z]+", english.lower()) if len(w) > 1 and w not in _STOP}
    return [
        g for g in vocabulary
        if g in CORE_VOCAB
        or any(part[:4].lower() in stems for part in g.split("-") if len(part) > 1 and part.lower() not in _STOP)
    ]


_STOP = set("the of a an and or to in on at for by with is am are was were be his her its my our their this that".split())


def _build_messages(english: str, vocabulary: list[str], topic: topics.Topic) -> list[BaseMessage]:
    have = set(vocabulary)
    topic_vocab = [w for w in topic.words if w in have]  # always offered, whatever the sentence says
    shown = set(topic_vocab)
    other = [g for g in _relevant_vocab(english, vocabulary) if g not in shown]
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT.format(
        topic=topic.label, context=topic.context,
        topic_vocabulary=" ".join(topic_vocab) or "(none loaded)",
        vocabulary=" ".join(other) or "(none)",
    ))]
    # general examples first, then the topic's own, which sit closest to the sentence
    for src, tgt in FEW_SHOT + list(topic.few_shot):
        messages += [HumanMessage(src), AIMessage(tgt)]
    messages.append(HumanMessage(english))
    return messages


def translate_node(state: GlossState) -> GlossState:
    topic = topics.get(state.get("topic"))
    response = _llm().invoke(_build_messages(state["english"], state.get("vocabulary", []), topic))
    return {"raw_output": str(response.content)}


def normalize_node(state: GlossState) -> GlossState:
    text = state["raw_output"].strip().upper()
    text = re.sub(r"^(ISL\s+)?GLOSS\s*:\s*", "", text)  # strip "GLOSS:" prefix if the model adds one
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"[^A-Z0-9\- ]+", " ", text)  # punctuation/quotes -> space, keep hyphens
    tokens = [t.strip("-") for t in text.split()]
    tokens = [t for t in tokens if t and t not in _DROP]
    return {"tokens": tokens, "gloss": " ".join(tokens)}


@lru_cache(maxsize=1)
def _graph():
    g = StateGraph(GlossState)
    g.add_node("translate", translate_node)
    g.add_node("normalize", normalize_node)
    g.add_edge(START, "translate")
    g.add_edge("translate", "normalize")
    g.add_edge("normalize", END)
    return g.compile()


def english_to_gloss(english: str, vocabulary: list[str], topic: str | None = None) -> list[str]:
    """Run the graph and return the gloss tokens in signing order. `topic` is a topics.TOPICS id."""
    if not english.strip():
        return []
    result = _graph().invoke({"english": english, "vocabulary": vocabulary, "topic": topics.get(topic).id})
    log.info("Gloss [%s]: %r -> %r", topics.get(topic).id, english, result["gloss"])
    return result["tokens"]
