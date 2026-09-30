"""Conversation topics for Phase 1 (Speech -> Sign): Casual, Hotel, Hospital.

A topic tells the pipeline where the conversation happens:
  - the gloss LLM gets the setting (so "check out" means leaving a hotel, "pressure" means blood
    pressure in a hospital), a few topic examples, and the topic's signs as preferred vocabulary;
  - Whisper gets the topic words as a hint, so "fever" or "towel" is heard correctly;
  - the UI shows topic phrases and the topic's signs first in the library.

`words` is every sign the topic should have. GET /topics reports which ones still lack a clip,
which is the to-do list for recording (scripts/import_clip.py).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    description: str
    context: str                          # one paragraph for the gloss prompt
    words: tuple[str, ...]                # glosses this topic needs, most important first
    examples: tuple[str, ...]             # quick phrases shown in the UI
    few_shot: tuple[tuple[str, str], ...] = ()  # (English, gloss) pairs added after the general examples


def _w(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text.split()))


CASUAL = Topic(
    id="casual",
    label="Casual",
    description="Everyday talk: greetings, introductions, family, feelings and plans.",
    context="An everyday conversation between friends, family or new acquaintances: greetings, "
            "introductions, family, feelings, work, study and plans.",
    words=_w(
        "HELLO BYE GOODBYE THANK-YOU SORRY PLEASE YES NO NOT FINE GOOD BAD "
        "GOOD-MORNING GOOD-AFTERNOON GOOD-EVENING GOOD-NIGHT HOW-ARE-YOU PLEASED NICE ALRIGHT "
        "MAN WOMAN STUDENT HOUSE SHOP "
        "I ME YOU YOUR HE SHE WE THEY OUR NAME AGE OLD YOUNG "
        "WHAT WHERE WHO WHY HOW WHEN WHICH "
        "FRIEND FAMILY MOTHER FATHER BROTHER SISTER SON DAUGHTER WIFE HUSBAND CHILD BABY "
        "HAPPY SAD LOVE LIKE KNOW UNDERSTAND LEARN WORK SCHOOL HOME COME GO SEE MEET TALK TELL ASK "
        "DEAF HEAR AGAIN SLOW FAST CAN CANNOT WANT NEED HELP "
        "TODAY TOMORROW YESTERDAY MORNING AFTERNOON EVENING NIGHT DAY WEEK TIME NOW SOON LATE "
        "EAT DRINK FOOD WATER SLEEP MARKET BUY TRAVEL CITY RAIN"
    ),
    examples=(
        "Hello, what is your name?",
        "Nice to meet you.",
        "How are you today?",
        "I don't understand, please say it again slowly.",
        "Where is your family?",
        "I am happy to see my friend.",
    ),
    few_shot=(
        ("Nice to meet you!", "YOU MEET HAPPY"),
        ("Please say that again slowly.", "AGAIN SLOW TELL PLEASE"),
        ("How old is your son?", "YOUR SON AGE WHAT"),
    ),
)

HOTEL = Topic(
    id="hotel",
    label="Hotel",
    description="Front desk, rooms, room service, payment and travel.",
    context="A guest talking with hotel staff (front desk, room service, housekeeping): booking, "
            "checking in and out, rooms, keys, beds, cleaning, food and meal times, payment, luggage, "
            "and travel to the station or airport. 'Check in' means ARRIVE, 'check out' means LEAVE.",
    words=_w(
        "HOTEL ROOM BED BEDROOM KEY DOOR LOCK WINDOW FLOOR LIFT TOILET BATHROOM TOWEL SOAP CLEAN DIRTY WASH CHANGE "
        "FAN LAMP TELEVISION TELEPHONE CHAIR TABLE RESTAURANT WAITER KITCHEN CARD PRICE MINUTE "
        "BUS PLANE TRAIN-STATION TRAIN-TICKET GOOD-MORNING GOOD-NIGHT "
        "HOT COLD WATER FOOD EAT DRINK MILK BREAD TEA COFFEE BREAKFAST LUNCH DINNER "
        "MONEY PAY COST HOW-MUCH CHEAP EXPENSIVE BILL MUCH MANY "
        "BOOK STAY ARRIVE LEAVE CHECK WAIT NIGHT DAY WEEK HOUR TIME MORNING EVENING TODAY TOMORROW EARLY LATE "
        "ONE TWO THREE FIVE PERSON PEOPLE FAMILY CHILD GUEST MANAGER SERVANT "
        "BAG BRING GIVE OPEN CLOSE CALL HELP NEED WANT PLEASE THANK-YOU SORRY "
        "TICKET CAR TAXI TRAIN TRAVEL DRIVE NEAR FAR HERE WHERE WHAT WHEN HOW"
    ),
    examples=(
        "I want a room for two nights.",
        "What time is breakfast?",
        "Please clean my room.",
        "How much does the room cost?",
        "The water is cold, please help.",
        "Please bring my bag to the room.",
        "I will leave tomorrow morning.",
    ),
    few_shot=(
        ("I want a room for two nights.", "I ROOM TWO NIGHT WANT"),
        ("How much does the room cost?", "ROOM COST HOW-MUCH"),
        ("I would like to check out now.", "NOW I LEAVE WANT"),
        ("The bathroom is dirty, please clean it.", "TOILET DIRTY CLEAN PLEASE"),
    ),
)

HOSPITAL = Topic(
    id="hospital",
    label="Hospital",
    description="Symptoms, pain, medicine, tests and emergencies.",
    context="A patient or family member talking with a doctor, nurse or hospital staff: symptoms, "
            "pain and where it hurts, body parts, how long it has lasted, medicine, tests, blood, "
            "pregnancy and emergencies. Medical meaning must stay exact: never change what hurts, "
            "where, how much, or for how long.",
    words=_w(
        "HOSPITAL DOCTOR NURSE PATIENT SICK HEALTHY PAIN HURT FEVER HOT COLD COUGH VOMIT WEAK TIRED SLEEP BREATHE "
        "MEDICINE TABLET INJECTION TEST BLOOD CHECK EMERGENCY AMBULANCE HELP "
        "HEAD STOMACH HEART CHEST BACK HAND ARM LEG EYE EAR NOSE MOUTH THROAT BONE BROKEN "
        "PREGNANT BABY BIRTH DIE HEAL BETTER WORSE "
        "DAY DAYS WEEK HOUR MINUTE TIME TODAY TOMORROW YESTERDAY NIGHT MORNING NOW SINCE "
        "I ME YOU MY YOUR HE SHE MOTHER FATHER CHILD WIFE HUSBAND "
        "WHERE WHAT WHEN HOW WHY WHO NEED WANT GIVE EAT DRINK WATER CAN CANNOT NOT YES NO PLEASE THANK-YOU"
    ),
    examples=(
        "I have a fever since yesterday.",
        "My stomach hurts.",
        "Where is the doctor?",
        "I cannot breathe, please help.",
        "Please give me medicine for the pain.",
        "My mother is sick and weak.",
        "When will the test be done?",
    ),
    few_shot=(
        ("I have had a fever for two days.", "I FEVER TWO DAY"),
        ("My stomach hurts a lot.", "MY STOMACH PAIN MUCH"),
        ("I cannot breathe, please help.", "I BREATHE CANNOT HELP PLEASE"),
        ("Please give me medicine for the pain.", "PAIN MEDICINE GIVE PLEASE"),
        ("Where is the doctor?", "DOCTOR WHERE"),
    ),
)

TOPICS: dict[str, Topic] = {t.id: t for t in (CASUAL, HOTEL, HOSPITAL)}
DEFAULT_TOPIC = CASUAL.id


def get(topic_id: str | None) -> Topic:
    """The topic with this id, or Casual for an unknown / empty id."""
    return TOPICS.get((topic_id or "").strip().lower(), TOPICS[DEFAULT_TOPIC])


def whisper_prompt(topic: Topic) -> str:
    """Short context for Whisper: the setting plus the topic's words, to bias recognition towards them."""
    words = ", ".join(w.lower().replace("-", " ") for w in topic.words[:60])
    return f"A conversation at a {topic.label.lower()}. Words: {words}."
