from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional


LETTERS = ("A", "B", "C", "D")
VALID_DIFFICULTIES = {"easy", "medium", "hard", "advanced"}


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", str(question)).strip().lower()


def question_similarity(left: str, right: str) -> float:
    """Return a conservative lexical similarity score from 0 to 1."""
    a = normalize_question(left)
    b = normalize_question(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    sequence_ratio = difflib.SequenceMatcher(a=a, b=b).ratio()
    tokens_a = set(re.findall(r"[a-z']+", a))
    tokens_b = set(re.findall(r"[a-z']+", b))
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    return max(sequence_ratio, jaccard)


def is_near_duplicate_question(
    question: str,
    recent_questions: Iterable[str],
    threshold: float = 0.86,
) -> bool:
    normalized = normalize_question(question)
    for recent in recent_questions:
        recent_normalized = normalize_question(recent)
        if normalized == recent_normalized:
            return True
        if question_similarity(normalized, recent_normalized) >= threshold:
            return True
    return False


def build_practice_signature(
    area: str,
    rule_id: str,
    pattern_id: str,
    question: str,
) -> str:
    normalized = normalize_question(question)
    return f"{area}|{rule_id}|{pattern_id}|{normalized}"


def _rotated_choices(
    correct: str,
    distractors: Iterable[str],
    seed: int,
) -> tuple[Dict[str, str], str]:
    values = [str(correct).strip()] + [str(value).strip() for value in distractors]
    if len(values) != 4 or len({value.lower() for value in values}) != 4:
        raise ValueError("Structured exercise choices must contain four unique values.")

    shift = seed % 4
    ordered = values[shift:] + values[:shift]
    choices = dict(zip(LETTERS, ordered))
    answer = next(letter for letter, value in choices.items() if value == values[0])
    return choices, answer


def _candidate(
    *,
    area: str,
    difficulty: str,
    rule_id: str,
    pattern_id: str,
    variant: int,
    question: str,
    correct: str,
    distractors: Iterable[str],
    explanation: str,
) -> Dict[str, Any]:
    choices, correct_answer = _rotated_choices(correct, distractors, variant)
    candidate_id = f"{area}:{difficulty}:{rule_id}:{pattern_id}:{variant}"
    return {
        "candidate_id": candidate_id,
        "grammar_area": area,
        "difficulty": difficulty,
        "rule_id": rule_id,
        "pattern_id": pattern_id,
        "question": question,
        "choices": choices,
        "correct_answer": correct_answer,
        "explanation": explanation,
        "source": "structured",
    }


def _past_tense() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    regular = [
        ("visit", "visited", "my grandparents"),
        ("clean", "cleaned", "the kitchen"),
        ("watch", "watched", "a documentary"),
        ("finish", "finished", "the report"),
        ("call", "called", "my cousin"),
        ("walk", "walked", "to the station"),
        ("paint", "painted", "the bedroom"),
        ("order", "ordered", "some noodles"),
    ]
    subjects = ["I", "we", "they", "my friends"]
    for i, (base, past, obj) in enumerate(regular * 2):
        subject = subjects[i % len(subjects)]
        out.append(_candidate(
            area="past_tense", difficulty="easy", rule_id="simple_past_regular",
            pattern_id=f"past_easy_{i % 4}", variant=i,
            question=f"Yesterday, {subject} ___ {obj}.", correct=past,
            distractors=[base, base + "s", base + "ing"],
            explanation="Use the simple past for a completed action at a finished past time.",
        ))

    irregular = [
        ("go", "went", "to the library"), ("buy", "bought", "a new charger"),
        ("take", "took", "the early train"), ("write", "wrote", "a long email"),
        ("see", "saw", "an old friend"), ("make", "made", "dinner"),
        ("leave", "left", "the office early"), ("find", "found", "my wallet"),
    ]
    times = ["Last night", "Last Saturday", "Two days ago", "Earlier this morning"]
    for i, (base, past, obj) in enumerate(irregular * 2):
        time = times[i % len(times)]
        out.append(_candidate(
            area="past_tense", difficulty="medium", rule_id="simple_past_irregular",
            pattern_id=f"past_medium_{i % 4}", variant=100 + i,
            question=f"{time}, she ___ {obj}.", correct=past,
            distractors=[base, base + "s", base + "ing"],
            explanation="Use the irregular simple past form for a completed past action.",
        ))

    hard_rows = [
        ("spoke", "At the conference last month, he ___ to the director after the meeting.", ["speak", "speaks", "speaking"]),
        ("chose", "After comparing both options, she ___ the quieter route home.", ["choose", "chooses", "choosing"]),
        ("drove", "During our holiday last year, my uncle ___ through the mountains at dawn.", ["drive", "drives", "driving"]),
        ("forgot", "Yesterday, he completely ___ the appointment.", ["forget", "forgets", "forgetting"]),
        ("taught", "Last semester, Professor Lee ___ the advanced class.", ["teach", "teaches", "teaching"]),
        ("caught", "After working late, she ___ the final bus home.", ["catch", "catches", "catching"]),
        ("brought", "At yesterday's meeting, they ___ all the documents.", ["bring", "brings", "bringing"]),
        ("thought", "Before accepting the job, I ___ about the offer for several days.", ["think", "thinks", "thinking"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard_rows * 2):
        out.append(_candidate(
            area="past_tense", difficulty="hard", rule_id="past_irregular_context",
            pattern_id=f"past_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the simple past because the event occurred in a completed past time period.",
        ))
    return out


def _present_tense() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = [
        ("work", "works", "from home on Fridays"), ("drink", "drinks", "tea every morning"),
        ("walk", "walks", "to school most days"), ("read", "reads", "before bed"),
        ("cook", "cooks", "dinner on Sundays"), ("exercise", "exercises", "after work"),
        ("study", "studies", "English every evening"), ("take", "takes", "the bus to work"),
    ]
    for i, (base, third, rest) in enumerate(rows * 2):
        out.append(_candidate(
            area="present_tense", difficulty="easy", rule_id="present_habit_third_person",
            pattern_id=f"present_easy_{i % 4}", variant=i,
            question=f"She ___ {rest}.", correct=third,
            distractors=[base, base + "ing", "did " + base],
            explanation="Use the simple present third person singular form for a regular habit with 'she'.",
        ))
    plural_rows = [
        ("work", "at the same company"), ("live", "near the coast"), ("meet", "on Tuesdays"),
        ("practice", "after class"), ("travel", "by train"), ("eat", "lunch together"),
        ("study", "in the library"), ("play", "tennis on weekends"),
    ]
    for i, (base, rest) in enumerate(plural_rows * 2):
        out.append(_candidate(
            area="present_tense", difficulty="medium", rule_id="present_habit_plural_subject",
            pattern_id=f"present_medium_{i % 4}", variant=100 + i,
            question=f"My friends usually ___ {rest}.", correct=base,
            distractors=[base + "s", base + "ing", "did " + base],
            explanation="Use the base form in the simple present with the plural subject 'my friends'.",
        ))
    facts = [
        ("Water", "boil", "boils", "at 100 degrees Celsius at sea level"),
        ("The Moon", "orbit", "orbits", "the Earth"),
        ("Fresh water", "freeze", "freezes", "at zero degrees Celsius"),
        ("A mirror", "reflect", "reflects", "light"),
        ("Air", "contain", "contains", "oxygen"),
        ("The Sun", "rise", "rises", "in the east"),
        ("A plant", "need", "needs", "water to survive"),
        ("A solar panel", "produce", "produces", "electricity from sunlight"),
    ]
    for i, (subject, base, third, rest) in enumerate(facts * 2):
        out.append(_candidate(
            area="present_tense", difficulty="hard", rule_id="general_truth_present",
            pattern_id=f"present_hard_{i % 4}", variant=200 + i,
            question=f"{subject} ___ {rest}.", correct=third,
            distractors=[base, base + "ing", "will " + base],
            explanation="Use the simple present for a general fact or truth.",
        ))
    return out


def _future_tense() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    predictions = [
        ("will rain", "I think it ___ this evening.", ["rained", "raining", "rain"]),
        ("will improve", "I think the situation ___ next month.", ["improved", "improving", "improve"]),
        ("will arrive", "I think the package ___ before noon.", ["arrived", "arriving", "arrive"]),
        ("will win", "I think our team ___ the match.", ["won", "winning", "win"]),
        ("will help", "I think Maya ___ with the move.", ["helped", "helping", "helps"]),
        ("will finish", "I think they ___ on time.", ["finished", "finishing", "finish"]),
        ("will call", "I think he ___ later tonight.", ["called", "calling", "calls"]),
        ("will recover", "I think she ___ soon.", ["recovered", "recovering", "recovers"]),
    ]
    for i, (correct, question, wrongs) in enumerate(predictions * 2):
        out.append(_candidate(
            area="future_tense", difficulty="easy", rule_id="will_prediction",
            pattern_id=f"future_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use 'will' plus the base verb for a prediction about the future.",
        ))
    going = [
        ("rain", "those dark clouds"), ("fall", "that unstable box"),
        ("overflow", "the water near the rim"), ("snow", "the heavy clouds"),
        ("crash", "that cyclist wobbling near the curb"), ("spill", "the tilted glass"),
        ("break", "the cracked shelf"), ("storm", "the lightning on the horizon"),
    ]
    for i, (verb, evidence) in enumerate(going * 2):
        out.append(_candidate(
            area="future_tense", difficulty="medium", rule_id="going_to_evidence",
            pattern_id=f"future_medium_{i % 4}", variant=100 + i,
            question=(f"Look at {evidence}. " + ("He ___ soon." if "cyclist" in evidence else "It ___ soon.")), correct=f"is going to {verb}",
            distractors=[f"will {verb}ed", f"{verb}ed", f"is {verb}"],
            explanation="Use 'be going to' for a future prediction based on present evidence.",
        ))
    perfect = [
        ("complete", "the report", "by Friday"), ("finish", "the course", "by June"),
        ("save", "enough money", "by next year"), ("read", "the whole book", "by tomorrow"),
        ("build", "the new bridge", "by 2030"), ("submit", "all applications", "by noon"),
        ("review", "every file", "by the deadline"), ("write", "ten chapters", "by December"),
    ]
    pps = {"complete":"completed","finish":"finished","save":"saved","read":"read","build":"built","submit":"submitted","review":"reviewed","write":"written"}
    for i, (verb, obj, deadline) in enumerate(perfect * 2):
        pp = pps[verb]
        out.append(_candidate(
            area="future_tense", difficulty="hard", rule_id="future_perfect_deadline",
            pattern_id=f"future_hard_{i % 4}", variant=200 + i,
            question=f"She ___ {obj} {deadline}.", correct=f"will have {pp}",
            distractors=[f"will {verb}", pp, f"has {pp}"],
            explanation="Use the future perfect for an action that will be completed before a future deadline.",
        ))
    return out


def _present_perfect() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pps = [
        ("lose", "lost", "my keys"), ("finish", "finished", "the assignment"),
        ("break", "broken", "her glasses"), ("forget", "forgotten", "the password"),
        ("complete", "completed", "the form"), ("send", "sent", "the email"),
        ("read", "read", "the article"), ("make", "made", "a decision"),
    ]
    for i, (base, pp, obj) in enumerate(pps * 2):
        out.append(_candidate(
            area="present_perfect", difficulty="easy", rule_id="present_result",
            pattern_id=f"perfect_easy_{i % 4}", variant=i,
            question=f"I ___ {obj} already.", correct=f"have {pp}",
            distractors=[base, base + "ing", f"had {pp}"],
            explanation="Use the present perfect for a completed action with relevance to the present.",
        ))
    duration = [
        ("live", "lived", "here", "five years"), ("work", "worked", "at this company", "three years"),
        ("know", "known", "her", "a decade"), ("study", "studied", "English", "two years"),
        ("own", "owned", "this car", "six years"), ("teach", "taught", "at the school", "four years"),
        ("use", "used", "this laptop", "three years"), ("wait", "waited", "for the reply", "two weeks"),
    ]
    for i, (base, pp, obj, duration_text) in enumerate(duration * 2):
        out.append(_candidate(
            area="present_perfect", difficulty="medium", rule_id="present_perfect_for_duration",
            pattern_id=f"perfect_medium_{i % 4}", variant=100 + i,
            question=f"We ___ {obj} for {duration_text}.", correct=f"have {pp}",
            distractors=[base, pp, f"are {base}ing"],
            explanation="Use the present perfect with 'for' when a situation began in the past and continues now.",
        ))
    experience = [
        ("visit", "visited", "Iceland"), ("see", "seen", "the northern lights"),
        ("ride", "ridden", "a horse"), ("eat", "eaten", "Korean barbecue"),
        ("fly", "flown", "in a helicopter"), ("meet", "met", "a famous author"),
        ("climb", "climbed", "a mountain"), ("sleep", "slept", "in a tent"),
    ]
    for i, (base, pp, obj) in enumerate(experience * 2):
        out.append(_candidate(
            area="present_perfect", difficulty="hard", rule_id="life_experience_ever",
            pattern_id=f"perfect_hard_{i % 4}", variant=200 + i,
            question=f"Have you ever ___ {obj}?", correct=pp,
            distractors=[base, base + "ing", "to " + base],
            explanation="Use the past participle after 'have' in a present perfect question about life experience.",
        ))
    return out


def _articles() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy_rows = [
        ("an", "She bought ___ apple at the market."),
        ("a", "He packed ___ banana for lunch."),
        ("an", "I saw ___ umbrella by the door."),
        ("a", "They found ___ camera under the seat."),
        ("an", "She peeled ___ orange after dinner."),
        ("a", "I need ___ notebook for class."),
        ("an", "We spoke with ___ engineer about the design."),
        ("a", "He called ___ doctor for advice."),
    ]
    for i, (article, question) in enumerate(easy_rows * 2):
        out.append(_candidate(
            area="articles", difficulty="easy", rule_id="indefinite_article_sound",
            pattern_id=f"articles_easy_{i % 4}", variant=i,
            question=question, correct=article,
            distractors=[value for value in ["a", "an", "the", "no article"] if value != article][:3],
            explanation="Choose 'a' or 'an' according to the first sound of the following word.",
        ))
    medium_rows = [
        ("moon", "the", "I could see ___ moon clearly tonight."),
        ("piano", "the", "She practices ___ piano every evening."),
        ("Louvre", "the", "We visited ___ Louvre in Paris."),
        ("sun", "the", "___ sun was bright this morning."),
        ("internet", "the", "I found the article on ___ internet."),
        ("guitar", "the", "He learned to play ___ guitar."),
        ("same answer", "the", "We both chose ___ same answer."),
        ("first chapter", "the", "Please read ___ first chapter tonight."),
    ]
    for i, (_, article, question) in enumerate(medium_rows * 2):
        out.append(_candidate(
            area="articles", difficulty="medium", rule_id="definite_article_conventional",
            pattern_id=f"articles_medium_{i % 4}", variant=100 + i,
            question=question, correct=article,
            distractors=["a", "an", "no article"],
            explanation="Use 'the' when English convention or the context identifies a specific or unique referent.",
        ))
    hard_rows = [
        ("bed", "no article", "After dinner, the children went to ___ bed."),
        ("president", "no article", "She was elected ___ president of the club."),
        ("advice", "no article", "He gave me ___ useful advice."),
        ("European", "a", "She is ___ European citizen."),
        ("university", "a", "He studies at ___ university near the river."),
        ("honest person", "an", "Everyone says he is ___ honest person."),
        ("school", "no article", "The children are at ___ school until three."),
        ("Mount Everest", "no article", "They hope to climb ___ Mount Everest one day."),
    ]
    for i, (_, article, question) in enumerate(hard_rows * 2):
        out.append(_candidate(
            area="articles", difficulty="hard", rule_id="article_exceptions_zero_and_sound",
            pattern_id=f"articles_hard_{i % 4}", variant=200 + i,
            question=question, correct=article,
            distractors=[value for value in ["a", "an", "the", "no article"] if value != article][:3],
            explanation="Choose the article according to English article conventions, including zero article and pronunciation based exceptions.",
        ))
    return out


def _prepositions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("in", "She was born ___ July."), ("on", "The keys are ___ the table."),
        ("at", "The meeting starts ___ nine o'clock."), ("on", "We have class ___ Monday."),
        ("in", "They live ___ Canada."), ("at", "I will meet you ___ the station."),
        ("under", "The cat is hiding ___ the chair."), ("between", "The bank is ___ the cafe and the pharmacy."),
    ]
    common = ["in", "on", "at", "by", "under", "between", "for", "with"]
    for i, (correct, question) in enumerate(easy * 2):
        distractors = [x for x in common if x != correct][:3]
        out.append(_candidate(
            area="prepositions", difficulty="easy", rule_id="basic_time_place_prepositions",
            pattern_id=f"prepositions_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=distractors,
            explanation="Choose the standard preposition used for this time or place expression.",
        ))
    medium = [
        ("in", "She is interested ___ modern art."), ("for", "This team is responsible ___ customer support."),
        ("of", "He is afraid ___ spiders."), ("to", "Please listen ___ the instructions."),
        ("with", "I agree ___ your main point."), ("about", "They complained ___ the noise."),
        ("for", "We apologized ___ the delay."), ("on", "The decision depends ___ the weather."),
    ]
    options = ["in", "for", "of", "to", "with", "about", "on", "at"]
    for i, (correct, question) in enumerate(medium * 2):
        distractors = [x for x in options if x != correct][i % 5:i % 5 + 3]
        if len(distractors) < 3:
            distractors = [x for x in options if x != correct][:3]
        out.append(_candidate(
            area="prepositions", difficulty="medium", rule_id="prepositional_collocations",
            pattern_id=f"prepositions_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=distractors,
            explanation="Use the preposition required by this common English collocation.",
        ))
    hard = [
        ("in", "She succeeded ___ persuading the committee."), ("into", "The report was divided ___ three sections."),
        ("from", "This material differs ___ the original."), ("with", "The policy is consistent ___ our guidelines."),
        ("of", "He was accused ___ hiding evidence."), ("to", "The committee objected ___ changing the deadline."),
        ("against", "They warned us ___ entering the area."), ("from", "Nothing can prevent her ___ applying."),
    ]
    options_hard = ["in", "into", "from", "with", "of", "to", "against", "on"]
    for i, (correct, question) in enumerate(hard * 2):
        distractors = [x for x in options_hard if x != correct][:3]
        out.append(_candidate(
            area="prepositions", difficulty="hard", rule_id="advanced_prepositional_collocations",
            pattern_id=f"prepositions_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=distractors,
            explanation="Use the preposition conventionally selected by this verb or adjective pattern.",
        ))
    return out


def _subject_verb_agreement() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("runs", "The dog ___ around the yard every morning.", ["run", "running", "did run"]),
        ("work", "My parents ___ at the same hospital.", ["works", "working", "did works"]),
        ("likes", "She ___ spicy food.", ["like", "liking", "did likes"]),
        ("study", "The students ___ in the library after class.", ["studies", "studying", "did studies"]),
        ("drives", "My brother ___ to work every day.", ["drive", "driving", "did drives"]),
        ("belong", "These books ___ to the school.", ["belongs", "belonging", "did belongs"]),
        ("explains", "The teacher ___ each rule clearly.", ["explain", "explaining", "did explains"]),
        ("need", "We ___ more time to finish.", ["needs", "needing", "did needs"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="subject_verb_agreement", difficulty="easy", rule_id="basic_subject_verb_number",
            pattern_id=f"sva_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="The verb must agree in number with the subject in the simple present.",
        ))
    medium = [
        ("is", "Each student ___ ready for the test.", ["are", "be", "being"]),
        ("is", "One of my friends ___ absent today.", ["are", "be", "being"]),
        ("is", "The list of names ___ complete.", ["are", "be", "being"]),
        ("are", "A number of employees ___ available.", ["is", "be", "being"]),
        ("is", "The number of applicants ___ surprising.", ["are", "be", "being"]),
        ("is", "Every book and notebook ___ labeled.", ["are", "be", "being"]),
        ("seems", "Neither answer ___ correct.", ["seem", "seeming", "seemed"]),
        ("has", "The quality of these products ___ improved.", ["have", "having", "had"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="subject_verb_agreement", difficulty="medium", rule_id="agreement_with_complex_subjects",
            pattern_id=f"sva_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Identify the grammatical head of the subject and make the finite verb agree with it.",
        ))
    hard = [
        ("are", "Either the manager or the assistants ___ attending the meeting.", ["is", "be", "being"]),
        ("is", "Either the assistants or the manager ___ attending the meeting.", ["are", "be", "being"]),
        ("is", "Neither the teachers nor the student ___ available.", ["are", "be", "being"]),
        ("are", "Neither the student nor the teachers ___ available.", ["is", "be", "being"]),
        ("is", "Ten dollars ___ enough for lunch.", ["are", "be", "being"]),
        ("is", "Three hours ___ enough time for the exam.", ["are", "be", "being"]),
        ("is", "The news ___ encouraging.", ["are", "be", "being"]),
        ("is", "Mathematics ___ my strongest subject.", ["are", "be", "being"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="subject_verb_agreement", difficulty="hard", rule_id="special_agreement_patterns",
            pattern_id=f"sva_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Apply the agreement rule for coordinated subjects, amounts, or nouns that look plural but act singular.",
        ))
    return out


def _verb_forms() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("run", "She can ___ very fast.", ["runs", "running", "to run"]),
        ("leave", "They must ___ now.", ["leaves", "leaving", "to leave"]),
        ("study", "You should ___ tonight.", ["studies", "studying", "to study"]),
        ("come", "He might ___ later.", ["comes", "coming", "to come"]),
        ("help", "We could ___ them tomorrow.", ["helps", "helping", "to help"]),
        ("call", "I will ___ you after work.", ["calls", "calling", "to call"]),
        ("rain", "It may ___ this afternoon.", ["rains", "raining", "to rain"]),
        ("wait", "You can ___ here.", ["waits", "waiting", "to wait"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="verb_forms", difficulty="easy", rule_id="base_form_after_modal",
            pattern_id=f"verbforms_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the base form of the verb directly after a modal verb.",
        ))
    medium = [
        ("working", "She is ___ from home today.", ["work", "worked", "to work"]),
        ("waiting", "They were ___ outside when I arrived.", ["wait", "waited", "to wait"]),
        ("finished", "He has ___ the report.", ["finish", "finishing", "to finish"]),
        ("written", "I had ___ the email before noon.", ["write", "wrote", "writing"]),
        ("broken", "The window was ___ by the storm.", ["break", "broke", "breaking"]),
        ("seen", "We have ___ that movie already.", ["see", "saw", "seeing"]),
        ("gone", "She has ___ home.", ["go", "went", "going"]),
        ("chosen", "They had ___ the cheaper option.", ["choose", "chose", "choosing"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="verb_forms", difficulty="medium", rule_id="auxiliary_required_form",
            pattern_id=f"verbforms_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the verb form required by the auxiliary in the construction.",
        ))
    hard = [
        ("finished", "Having ___ the report, she left the office.", ["finish", "finishing", "finishes"]),
        ("revised", "The proposal needs to be ___ before Friday.", ["revise", "revising", "revises"]),
        ("warned", "Having been ___ about the storm, they changed course.", ["warn", "warning", "warns"]),
        ("known", "You should have ___ the answer.", ["know", "knew", "knowing"]),
        ("gone", "They could have ___ earlier.", ["go", "went", "going"]),
        ("considered", "The proposal is being ___ by the committee.", ["consider", "considering", "considers"]),
        ("completed", "She appears to have ___ the task.", ["complete", "completing", "completes"]),
        ("waiting", "They had been ___ for over an hour.", ["wait", "waited", "to wait"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="verb_forms", difficulty="hard", rule_id="complex_nonfinite_and_perfect_forms",
            pattern_id=f"verbforms_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the verb form required by the perfect, passive, or nonfinite construction.",
        ))
    return out


def _adjectives() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pairs = [
        ("boring", "bored", "movie", "audience"), ("interesting", "interested", "lecture", "students"),
        ("exciting", "excited", "game", "fans"), ("confusing", "confused", "manual", "readers"),
        ("surprising", "surprised", "result", "researchers"), ("tiring", "tired", "journey", "travelers"),
        ("frightening", "frightened", "noise", "children"), ("amusing", "amused", "story", "listeners"),
    ]
    for i, (cause, feeling, thing, people) in enumerate(pairs * 2):
        out.append(_candidate(
            area="adjectives", difficulty="easy", rule_id="ed_ing_adjectives_cause",
            pattern_id=f"adjectives_easy_{i % 4}", variant=i,
            question=f"The {thing} was very ___.", correct=cause,
            distractors=[feeling, cause + "ly", cause.rstrip("ing")],
            explanation="Use the -ing adjective to describe something that causes the feeling.",
        ))
    for i, (cause, feeling, thing, people) in enumerate(pairs * 2):
        out.append(_candidate(
            area="adjectives", difficulty="medium", rule_id="ed_ing_adjectives_feeling",
            pattern_id=f"adjectives_medium_{i % 4}", variant=100 + i,
            question=f"The {people} felt ___.", correct=feeling,
            distractors=[cause, feeling + "ly", feeling.rstrip("ed")],
            explanation="Use the -ed adjective to describe how a person or group feels.",
        ))
    hard_rows = [
        ("convincing", "The committee found the proposal ___.", ["convincingly", "convince", "to convince"]),
        ("reliable", "The data appears ___.", ["reliably", "reliability", "rely"]),
        ("unlikely", "Such an outcome seems ___.", ["unlikelyly", "unlikeliness", "unlike"]),
        ("valuable", "The experience proved ___.", ["valuably", "value", "valuation"]),
        ("effective", "The new method became ___.", ["effectively", "effect", "effecting"]),
        ("essential", "Clear communication remains ___.", ["essentially", "essence", "essentiality"]),
        ("possible", "A compromise is still ___.", ["possibly", "possibility", "possiblely"]),
        ("appropriate", "The response sounded ___.", ["appropriately", "appropriateness", "appropriateing"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard_rows * 2):
        out.append(_candidate(
            area="adjectives", difficulty="hard", rule_id="adjective_after_linking_or_complex_verb",
            pattern_id=f"adjectives_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use an adjective as the complement after this linking or complex verb.",
        ))
    return out


def _adverbs() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pairs = [
        ("careful", "carefully", "drives"), ("quiet", "quietly", "speaks"),
        ("slow", "slowly", "walks"), ("clear", "clearly", "explains"),
        ("patient", "patiently", "waits"), ("polite", "politely", "answers"),
        ("quick", "quickly", "responds"), ("beautiful", "beautifully", "sings"),
    ]
    for i, (adj, adv, verb) in enumerate(pairs * 2):
        out.append(_candidate(
            area="adverbs", difficulty="easy", rule_id="adverb_modifies_action",
            pattern_id=f"adverbs_easy_{i % 4}", variant=i,
            question=f"She {verb} ___.", correct=adv,
            distractors=[adj, adj + "ness", adv + "er"],
            explanation="Use an adverb to describe how the action is performed.",
        ))
    medium = [
        ("well", "He plays the piano ___.", ["good", "goodly", "betterly"]),
        ("hard", "They worked ___ all afternoon.", ["hardly", "hardness", "harderly"]),
        ("fast", "The train moved ___.", ["fastly", "fasterly", "fastness"]),
        ("late", "She arrived ___ for the meeting.", ["lately", "lateness", "laterly"]),
        ("carefully", "Please read the contract ___.", ["careful", "carefulness", "carefuller"]),
        ("efficiently", "The team completed the task ___.", ["efficient", "efficiency", "efficientness"]),
        ("calmly", "He responded ___.", ["calm", "calmness", "calming"]),
        ("accurately", "The machine measures temperature ___.", ["accurate", "accuracy", "accurateness"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="adverbs", difficulty="medium", rule_id="adverb_form_and_irregular_adverbs",
            pattern_id=f"adverbs_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the correct adverb form to modify the verb in this sentence.",
        ))
    hard = [
        ("barely", "She had ___ finished when the power went out.", ["bare", "bareness", "baring"]),
        ("significantly", "Sales have increased ___ this year.", ["significant", "significance", "signify"]),
        ("considerably", "The situation has improved ___.", ["considerable", "consideration", "consider"]),
        ("remarkably", "The device performed ___ well.", ["remarkable", "remark", "remarking"]),
        ("virtually", "The two versions are ___ identical.", ["virtual", "virtue", "virtuality"]),
        ("highly", "The proposal is ___ controversial.", ["high", "height", "higher"]),
        ("closely", "The researchers worked ___ together.", ["close", "closeness", "closer"]),
        ("widely", "The theory is ___ accepted.", ["wide", "width", "widen"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="adverbs", difficulty="hard", rule_id="advanced_adverb_collocation",
            pattern_id=f"adverbs_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the adverb that correctly modifies the following verb, adjective, or adverb phrase.",
        ))
    return out


def _pronouns() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("me", "Sarah called ___ yesterday.", ["I", "my", "mine"]),
        ("him", "I saw ___ at the station.", ["he", "his", "himself's"]),
        ("her", "We invited ___ to dinner.", ["she", "hers", "her's"]),
        ("us", "The teacher gave ___ extra time.", ["we", "our", "ours"]),
        ("them", "I helped ___ with the boxes.", ["they", "their", "theirs"]),
        ("me", "Please send ___ the file.", ["I", "mine", "my"]),
        ("him", "The coach praised ___.", ["he", "his", "himself's"]),
        ("her", "They asked ___ a question.", ["she", "hers", "her's"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="pronouns", difficulty="easy", rule_id="object_pronouns",
            pattern_id=f"pronouns_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use an object pronoun when the pronoun is the object of the verb.",
        ))
    medium = [
        ("me", "Between you and ___, this plan worries me.", ["I", "my", "mine"]),
        ("him", "The package is for ___.", ["he", "his", "himself's"]),
        ("her", "I sat beside ___.", ["she", "hers", "her's"]),
        ("us", "They spoke to ___ after class.", ["we", "our", "ours"]),
        ("them", "The decision depends on ___.", ["they", "their", "theirs"]),
        ("me", "She came with ___.", ["I", "my", "mine"]),
        ("him", "No one except ___ knew the answer.", ["he", "his", "himself's"]),
        ("her", "The gift came from ___.", ["she", "hers", "her's"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="pronouns", difficulty="medium", rule_id="object_pronouns_after_prepositions",
            pattern_id=f"pronouns_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the object form of the pronoun after a preposition.",
        ))
    hard = [
        ("who", "The person ___ called left no message.", ["whom", "whose", "which"]),
        ("whom", "The person to ___ I spoke was helpful.", ["who", "whose", "which"]),
        ("whom", "The students, all of ___ passed, celebrated.", ["who", "they", "which"]),
        ("who", "Anyone ___ wants to join may sign up.", ["whom", "whose", "which"]),
        ("whom", "The candidate ___ we interviewed was excellent.", ["whoever", "whose", "which"]),
        ("whose", "The author ___ book won the prize thanked her editor.", ["who", "whom", "which"]),
        ("who", "The engineers ___ designed the bridge received awards.", ["whom", "whose", "which"]),
        ("whom", "The colleague with ___ I traveled speaks French.", ["who", "whose", "which"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="pronouns", difficulty="hard", rule_id="relative_pronoun_case",
            pattern_id=f"pronouns_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the relative pronoun whose case and function fit the clause.",
        ))
    return out


def _conjunctions() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("so", "I was tired, ___ I went to bed early.", ["or", "although", "unless"]),
        ("or", "Would you like tea ___ coffee?", ["because", "so", "although"]),
        ("but", "The task was difficult, ___ we finished it.", ["because", "so", "or"]),
        ("and", "She opened the window ___ turned on the fan.", ["because", "although", "unless"]),
        ("because", "I stayed home ___ I was sick.", ["but", "or", "so"]),
        ("so", "It was raining, ___ we took an umbrella.", ["although", "or", "because"]),
        ("but", "He is young, ___ he is very experienced.", ["because", "so", "or"]),
        ("or", "Call me ___ send me a message.", ["because", "although", "so"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="conjunctions", difficulty="easy", rule_id="basic_clause_conjunctions",
            pattern_id=f"conjunctions_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the conjunction that expresses the intended relationship between the ideas.",
        ))
    medium = [
        ("Although", "___ it was raining, we went hiking.", ["Because", "So", "Unless"]),
        ("because", "She left early ___ she felt unwell.", ["although", "but", "or"]),
        ("unless", "You cannot enter ___ you have a ticket.", ["although", "because", "so"]),
        ("while", "I cooked ___ he set the table.", ["because", "unless", "so"]),
        ("whereas", "My brother loves cities, ___ I prefer the countryside.", ["because", "so", "unless"]),
        ("Since", "___ everyone is here, we can begin.", ["Although", "Or", "Unless"]),
        ("Even though", "___ she was exhausted, she kept working.", ["Because", "So that", "Unless"]),
        ("so that", "Speak clearly ___ everyone can hear you.", ["although", "because of", "unless"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="conjunctions", difficulty="medium", rule_id="subordinating_conjunction_meaning",
            pattern_id=f"conjunctions_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the subordinating conjunction that matches the relationship between the clauses.",
        ))
    hard = [
        ("provided that", "You may borrow the car ___ you return it by ten.", ["whereas", "even though", "because of"]),
        ("in case", "Take an umbrella ___ it rains later.", ["whereas", "although", "so that not"]),
        ("as long as", "You can stay here ___ you keep the room tidy.", ["whereas", "even though", "because of"]),
        ("whereas", "The first method is cheap, ___ the second is faster.", ["because", "so that", "unless"]),
        ("even though", "He continued ___ he knew the risks.", ["provided that", "so that", "because of"]),
        ("so that", "She whispered ___ she would not wake the baby.", ["whereas", "unless", "even though"]),
        ("Once", "___ the payment arrives, we will ship the order.", ["Although", "Whereas", "Because of"]),
        ("whether", "I do not know ___ he will attend.", ["because", "so", "whereas"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="conjunctions", difficulty="hard", rule_id="advanced_clause_linkers",
            pattern_id=f"conjunctions_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the clause linker that precisely expresses the intended condition, contrast, purpose, or uncertainty.",
        ))
    return out


def _conditionals() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("will stay", "If it rains tomorrow, we ___ inside.", ["stayed", "would stayed", "staying"]),
        ("melts", "If you heat ice, it ___.", ["will melted", "would melt yesterday", "melting"]),
        ("will call", "If I finish early, I ___ you.", ["called", "would called", "calling"]),
        ("boils", "If water reaches 100°C at sea level, it ___.", ["will boiled", "would boil yesterday", "boiling"]),
        ("will help", "If you ask politely, she ___ you.", ["helped", "would helped", "helping"]),
        ("rusts", "If iron gets wet repeatedly, it ___.", ["will rusted", "would rust yesterday", "rusting"]),
        ("will open", "If you press this button, the door ___.", ["opened yesterday", "would opened", "opening"]),
        ("evaporates", "If water is heated long enough, it ___.", ["will evaporated", "would evaporate yesterday", "evaporating"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="conditionals", difficulty="easy", rule_id="zero_and_first_conditionals",
            pattern_id=f"conditionals_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the tense pattern required by a zero or first conditional.",
        ))
    medium = [
        ("would study", "If I had more time, I ___ Spanish.", ["will study", "studied yesterday", "studying"]),
        ("would travel", "If she had more money, she ___ more often.", ["will travel", "traveled yesterday", "traveling"]),
        ("would buy", "If we lived closer, we ___ that house.", ["will buy", "bought yesterday", "buying"]),
        ("would accept", "If I were you, I ___ the offer.", ["will accepted", "accepted yesterday", "accepting"]),
        ("would feel", "If he slept more, he ___ better.", ["will felt", "felt yesterday", "feeling"]),
        ("would join", "If they had space, they ___ us.", ["will joined", "joined yesterday", "joining"]),
        ("would choose", "If she knew the answer, she ___ it.", ["will choose", "chose yesterday", "choosing"]),
        ("would move", "If I worked remotely, I ___ abroad.", ["will moved", "moved yesterday", "moving"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="conditionals", difficulty="medium", rule_id="second_conditional",
            pattern_id=f"conditionals_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use 'would' plus the base verb in the result clause of a second conditional.",
        ))
    hard = [
        ("would have caught", "If they had left earlier, they ___ the train.", ["will catch", "would caught", "catching"]),
        ("would have passed", "If he had studied more, he ___ the exam.", ["will pass", "would passed", "passing"]),
        ("would have arrived", "If we had taken a taxi, we ___ sooner.", ["will arrive", "would arrived", "arriving"]),
        ("would have known", "If she had read the message, she ___ about the change.", ["will know", "would knew", "knowing"]),
        ("would have saved", "If I had backed up the file, I ___ the data.", ["will save", "would saved", "saving"]),
        ("would have avoided", "If they had checked the map, they ___ the traffic.", ["will avoid", "would avoided", "avoiding"]),
        ("would have won", "If the team had scored once more, it ___ the match.", ["will win", "would won", "winning"]),
        ("would have helped", "If you had called me, I ___ you.", ["will help", "would helped", "helping"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="conditionals", difficulty="hard", rule_id="third_conditional",
            pattern_id=f"conditionals_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use 'would have' plus a past participle in the result clause of a third conditional.",
        ))
    return out


def _modals() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("must", "You ___ wear a seat belt.", ["must to", "musts", "musting"]),
        ("can", "She ___ swim very well.", ["can to", "cans", "canning"]),
        ("may", "You ___ leave early today.", ["may to", "mays", "maying"]),
        ("should", "We ___ check the address first.", ["should to", "shoulds", "shoulding"]),
        ("might", "It ___ rain later.", ["might to", "mights", "mighting"]),
        ("could", "He ___ help us tomorrow.", ["could to", "coulds", "coulding"]),
        ("would", "I ___ prefer tea.", ["would to", "woulds", "woulding"]),
        ("must", "Visitors ___ sign in.", ["must to", "musts", "musting"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="modals", difficulty="easy", rule_id="basic_modal_form",
            pattern_id=f"modals_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the modal verb without an infinitive marker or agreement ending.",
        ))
    medium = [
        ("come", "She might ___ later.", ["comes", "to come", "coming"]),
        ("finish", "You must ___ this today.", ["finishes", "to finish", "finishing"]),
        ("wait", "We should ___ a little longer.", ["waits", "to wait", "waiting"]),
        ("call", "He could ___ the office.", ["calls", "to call", "calling"]),
        ("leave", "They may ___ after lunch.", ["leaves", "to leave", "leaving"]),
        ("try", "You should ___ again.", ["tries", "to try", "trying"]),
        ("ask", "We could ___ for help.", ["asks", "to ask", "asking"]),
        ("bring", "You must ___ your passport.", ["brings", "to bring", "bringing"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="modals", difficulty="medium", rule_id="modal_plus_base_verb",
            pattern_id=f"modals_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the base form directly after a modal verb.",
        ))
    hard = [
        ("have told", "You should ___ me earlier.", ["to have told", "have telling", "has told"]),
        ("have left", "They might ___ already.", ["to have left", "have leaving", "has left"]),
        ("have known", "She must ___ the answer.", ["to have known", "have knowing", "has knew"]),
        ("leave", "You had better ___ now.", ["to leave", "leaves", "leaving"]),
        ("have arrived", "The train should ___ by now.", ["to have arrived", "have arriving", "has arrived"]),
        ("have forgotten", "He may ___ the meeting.", ["to have forgotten", "have forgetting", "has forgot"]),
        ("have been", "They could ___ delayed.", ["to have been", "have being", "has been"]),
        ("have checked", "We should ___ the figures again.", ["to have checked", "have checking", "has checked"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="modals", difficulty="hard", rule_id="modal_perfect_and_semi_modal",
            pattern_id=f"modals_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the correct base or perfect form after the modal or semi modal expression.",
        ))
    return out


def _sentence_structure() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("we stayed inside", "Because it was raining, ___.", ["because inside", "raining outside", "stayed"]),
        ("I went to bed", "After I finished my homework, ___.", ["because tired", "finishing", "to bed"]),
        ("she called her mother", "When the train arrived, ___.", ["because the train", "calling", "at the station"]),
        ("they started dinner", "Once everyone was home, ___.", ["because hungry", "starting", "at home"]),
        ("we opened the windows", "Since the room was hot, ___.", ["because heat", "opening", "the windows"]),
        ("he took a taxi", "Because he was late, ___.", ["because traffic", "taking", "to work"]),
        ("the class began", "After the teacher arrived, ___.", ["because class", "beginning", "in class"]),
        ("I turned on the light", "When it became dark, ___.", ["because dark", "turning", "the light"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="sentence_structure", difficulty="easy", rule_id="dependent_plus_independent_clause",
            pattern_id=f"structure_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="A dependent clause must connect to a complete independent clause.",
        ))
    medium = [
        ("is a doctor", "The woman who lives next door ___.", ["because a doctor", "in the garden", "living nearby"]),
        ("won the prize", "The book that I recommended ___.", ["because popular", "on the shelf", "winning"]),
        ("needs repair", "The computer on my desk ___.", ["because old", "under the lamp", "repairing"]),
        ("arrived late", "The students from the evening class ___.", ["because traffic", "in the hall", "arriving"]),
        ("was expensive", "The hotel we booked ___.", ["because central", "near the station", "costing"]),
        ("belongs to my sister", "The bicycle by the gate ___.", ["because blue", "near the wall", "belonging"]),
        ("has closed", "The shop across the street ___.", ["because quiet", "on the corner", "closing"]),
        ("starts tomorrow", "The course I registered for ___.", ["because useful", "at the college", "starting"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="sentence_structure", difficulty="medium", rule_id="complete_main_clause_after_complex_subject",
            pattern_id=f"structure_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="A complex subject still needs a complete finite predicate to form a sentence.",
        ))
    hard = [
        ("she emailed it to her manager", "Having finished the report, ___.", ["the office was quiet", "there was an email", "the deadline arrived"]),
        ("he went straight to bed", "Exhausted after the flight, ___.", ["the hotel was quiet", "there was a bed", "the night was long"]),
        ("the researchers repeated the experiment", "Unsure of the result, ___.", ["the result was unclear", "there was uncertainty", "the laboratory closed"]),
        ("she checked every figure again", "Concerned about an error, ___.", ["the spreadsheet was large", "there was a mistake", "the deadline approached"]),
        ("they postponed the launch", "Faced with a serious defect, ___.", ["the product was new", "there was a defect", "the market changed"]),
        ("he asked for clarification", "Confused by the instructions, ___.", ["the instructions were long", "there was confusion", "the meeting ended"]),
        ("we changed our route", "Warned about the storm, ___.", ["the storm grew stronger", "there was rain", "the road was narrow"]),
        ("she revised the introduction", "Dissatisfied with the opening paragraph, ___.", ["the paragraph was short", "there was a draft", "the editor called"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="sentence_structure", difficulty="hard", rule_id="logical_subject_of_participial_modifier",
            pattern_id=f"structure_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="The understood subject of the opening modifier must logically match the subject of the main clause.",
        ))
    return out


def _word_order() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("always drinks coffee", "She ___ before work.", ["drinks always coffee", "coffee always drinks", "always coffee drinks"]),
        ("usually reads books", "He ___ in the evening.", ["reads usually books", "books usually reads", "usually books reads"]),
        ("often takes the bus", "My sister ___ to work.", ["takes often the bus", "the bus often takes", "often the bus takes"]),
        ("sometimes cooks dinner", "He ___ on Fridays.", ["cooks sometimes dinner", "dinner sometimes cooks", "sometimes dinner cooks"]),
        ("rarely watches television", "She ___ after midnight.", ["watches rarely television", "television rarely watches", "rarely television watches"]),
        ("usually arrives early", "Our teacher ___ for class.", ["arrives usually early", "early usually arrives", "usually early arrives"]),
        ("always checks the door", "He ___ before leaving.", ["checks always the door", "the door always checks", "always the door checks"]),
        ("often visits her family", "She ___ on weekends.", ["visits often her family", "her family often visits", "often her family visits"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="word_order", difficulty="easy", rule_id="frequency_adverb_position",
            pattern_id=f"wordorder_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Frequency adverbs normally come before the main verb in this structure.",
        ))
    embedded = [
        ("where he lives", "I don't know ___.", ["where does he live", "where he does live", "where lives he"]),
        ("what time the store closes", "Can you tell me ___?", ["what time does the store close", "what time closes the store", "does the store close what time"]),
        ("why she left", "Do you know ___?", ["why did she leave", "why she did leave", "why left she"]),
        ("when the train arrives", "Could you tell me ___?", ["when does the train arrive", "when arrives the train", "does the train arrive when"]),
        ("how much it costs", "I wonder ___.", ["how much does it cost", "how much costs it", "does it cost how much"]),
        ("where they went", "Nobody knows ___.", ["where did they go", "where went they", "did they go where"]),
        ("who she invited", "I can't remember ___.", ["who did she invite", "who she did invite", "did she invite who"]),
        ("whether he is coming", "Please find out ___.", ["whether is he coming", "is he coming whether", "whether does he come"]),
    ]
    for i, (correct, question, wrongs) in enumerate(embedded * 2):
        out.append(_candidate(
            area="word_order", difficulty="medium", rule_id="embedded_question_statement_order",
            pattern_id=f"wordorder_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use statement word order inside an embedded question.",
        ))
    inversion = [
        ("did I understand", "Only then ___ the problem.", ["I did understand", "I understood did", "did understand I"]),
        ("had I seen", "Never before ___ such a view.", ["I had seen", "I seen had", "had seen I"]),
        ("did she realize", "Only later ___ the mistake.", ["she did realize", "she realized did", "did realize she"]),
        ("have we faced", "Rarely ___ such a difficult choice.", ["we have faced", "we faced have", "have faced we"]),
        ("did they know", "Little ___ what would happen next.", ["they did know", "they knew did", "did know they"]),
        ("was the room", "So quiet ___ that we could hear the clock.", ["the room was", "the room did be", "was quiet the room"]),
        ("had he arrived", "Hardly ___ when the meeting began.", ["he had arrived", "he arrived had", "had arrived he"]),
        ("can we solve", "Only by working together ___ this problem.", ["we can solve", "we solve can", "can solve we"]),
    ]
    for i, (correct, question, wrongs) in enumerate(inversion * 2):
        out.append(_candidate(
            area="word_order", difficulty="hard", rule_id="inversion_after_fronted_expression",
            pattern_id=f"wordorder_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use subject auxiliary inversion after the fronted negative or restrictive expression.",
        ))
    return out


def _countable_uncountable() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("information", "I need some ___ about the course.", ["informations", "an information", "one information"]),
        ("furniture", "We bought new ___ for the office.", ["furnitures", "a furniture", "one furniture"]),
        ("advice", "She gave me useful ___.", ["advices", "an advice", "one advice"]),
        ("equipment", "The lab needs more ___.", ["equipments", "an equipment", "one equipment"]),
        ("luggage", "There isn't much ___ in the car.", ["luggages", "a luggage", "one luggage"]),
        ("homework", "The teacher gave us some ___.", ["homeworks", "a homework", "one homework"]),
        ("research", "We need more ___ before deciding.", ["researches", "a research", "one research"]),
        ("traffic", "There was heavy ___ this morning.", ["traffics", "a traffic", "one traffic"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="countable_uncountable_nouns", difficulty="easy", rule_id="basic_uncountable_nouns",
            pattern_id=f"countability_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the uncountable noun without an indefinite article or plural ending.",
        ))
    units = [
        ("pieces of advice", "She gave me three ___ before the interview.", ["advices", "advice", "an advice"]),
        ("pieces of furniture", "We ordered three ___ for the living room.", ["furnitures", "furniture", "a furniture"]),
        ("pieces of information", "The form asks for three ___ about your work history.", ["informations", "information", "an information"]),
        ("pieces of equipment", "The lab purchased three new ___ this year.", ["equipments", "equipment", "an equipment"]),
        ("pieces of luggage", "The airline allows three ___ for this ticket.", ["luggages", "luggage", "a luggage"]),
        ("loaves of bread", "The bakery delivered three ___ this morning.", ["breads", "bread", "a bread"]),
        ("sheets of paper", "Please give me three ___ for the printer.", ["papers", "paper", "a paper"]),
        ("pieces of work", "The exhibition includes three early ___ by the artist.", ["works of work", "work", "a work"]),
    ]
    for i, (correct, question, wrongs) in enumerate(units * 2):
        out.append(_candidate(
            area="countable_uncountable_nouns", difficulty="medium", rule_id="countable_units_for_mass_nouns",
            pattern_id=f"countability_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use a countable unit expression when giving an exact number with an uncountable noun.",
        ))
    quantifiers = [
        ("much", "There isn't ___ information available.", ["many", "a few", "several"]),
        ("many", "There aren't ___ chairs left.", ["much", "a little", "less"]),
        ("a little", "We have ___ time before the train.", ["a few", "many", "several"]),
        ("a few", "She asked ___ questions.", ["a little", "much", "less"]),
        ("less", "We need ___ equipment this year.", ["fewer", "many", "a few"]),
        ("fewer", "There are ___ mistakes in this version.", ["less", "much", "a little"]),
        ("much", "How ___ luggage did you bring?", ["many", "few", "several"]),
        ("many", "How ___ applicants were interviewed?", ["much", "little", "less"]),
    ]
    for i, (correct, question, wrongs) in enumerate(quantifiers * 2):
        out.append(_candidate(
            area="countable_uncountable_nouns", difficulty="hard", rule_id="quantifiers_by_countability",
            pattern_id=f"countability_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Choose the quantifier that matches whether the noun is countable or uncountable.",
        ))
    return out


def _plurals() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("children", "Two ___ are playing outside.", ["child", "childs", "childrens"]),
        ("teeth", "I brushed my ___ before bed.", ["tooth", "tooths", "teeths"]),
        ("feet", "My ___ hurt after the long walk.", ["foot", "foots", "feets"]),
        ("mice", "The scientist observed several ___.", ["mouse", "mouses", "mices"]),
        ("geese", "We saw three ___ near the lake.", ["goose", "gooses", "geeses"]),
        ("men", "Two ___ carried the table.", ["man", "mans", "mens"]),
        ("women", "Several ___ spoke at the conference.", ["woman", "womans", "womens"]),
        ("people", "Many ___ attended the festival.", ["person", "persons one", "peopleses"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="plurals", difficulty="easy", rule_id="irregular_plural_nouns",
            pattern_id=f"plurals_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the correct irregular plural form in this plural context.",
        ))
    medium = [
        ("sheep", "The farmer keeps five ___.", ["sheeps", "sheepes", "a sheep"]),
        ("deer", "We saw several ___ in the forest.", ["deers", "deeres", "a deer"]),
        ("species", "The island contains several rare ___.", ["specieses", "specie", "a species"]),
        ("series", "The platform released two new ___.", ["serieses", "serie", "a series"]),
        ("aircraft", "The airline owns twenty ___.", ["aircrafts", "aircraftes", "an aircraft"]),
        ("fish", "The aquarium contains hundreds of ___.", ["fisheses", "fishs", "a fish"]),
        ("offspring", "The animals produced several ___.", ["offsprings", "offspringes", "an offspring"]),
        ("means", "Several ___ of transport are available.", ["meanses", "mean", "a means"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="plurals", difficulty="medium", rule_id="invariant_plural_nouns",
            pattern_id=f"plurals_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Some nouns keep the same form in both the singular and the plural.",
        ))
    hard = [
        ("knives", "The chef sharpened several ___.", ["knifes", "knife", "knivies"]),
        ("leaves", "The tree lost many ___ in the storm.", ["leafs", "leaf", "leafes"]),
        ("lives", "The new treatment has saved many ___.", ["lifes", "life", "lifees"]),
        ("wives", "The novel follows the lives of three ___.", ["wifes", "wife", "wifees"]),
        ("analyses", "The report contains several statistical ___.", ["analysises", "analysis", "analysisies"]),
        ("crises", "The country has faced several economic ___.", ["crisises", "crisis", "crisisies"]),
        ("criteria", "The committee uses five ___ to evaluate applications.", ["criterions", "criterion", "criterias"]),
        ("phenomena", "The course examines unusual natural ___.", ["phenomenons", "phenomenon", "phenomenas"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="plurals", difficulty="hard", rule_id="advanced_plural_forms",
            pattern_id=f"plurals_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the conventional plural form of this noun.",
        ))
    return out


def _gerunds_infinitives() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    gerund = [
        ("reading", "I enjoy ___ before bed.", ["to read", "read", "reads"]),
        ("driving", "She avoids ___ in heavy traffic.", ["to drive", "drive", "drives"]),
        ("leaving", "They suggested ___ earlier.", ["to leave", "leave", "leaves"]),
        ("moving", "We are considering ___ to a larger apartment.", ["to move", "move", "moves"]),
        ("writing", "He finished ___ the report before lunch.", ["to write", "write", "writes"]),
        ("waiting", "Do you mind ___ a few minutes?", ["to wait", "wait", "waits"]),
        ("trying", "She kept ___ until the program worked.", ["to try", "try", "tries"]),
        ("speaking", "They practice ___ English every day.", ["to speak", "speak", "speaks"]),
    ]
    for i, (correct, question, wrongs) in enumerate(gerund * 2):
        out.append(_candidate(
            area="gerunds_infinitives", difficulty="easy", rule_id="gerund_selecting_verbs",
            pattern_id=f"gerunds_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="The governing verb in this sentence is followed by a gerund.",
        ))
    infinitive = [
        ("to leave", "She decided ___ early.", ["leaving", "leave", "will leave"]),
        ("to travel", "They hope ___ next summer.", ["traveling", "travel", "will travel"]),
        ("to study", "We plan ___ abroad next year.", ["studying", "study", "will study"]),
        ("to improve", "I want ___ my pronunciation.", ["improving", "improve", "will improve"]),
        ("to help", "He agreed ___ with the project.", ["helping", "help", "will help"]),
        ("to answer", "She refused ___ the question.", ["answering", "answer", "will answer"]),
        ("to finish", "They managed ___ before the deadline.", ["finishing", "finish", "will finish"]),
        ("to call", "He promised ___ after work.", ["calling", "call", "will call"]),
    ]
    for i, (correct, question, wrongs) in enumerate(infinitive * 2):
        out.append(_candidate(
            area="gerunds_infinitives", difficulty="medium", rule_id="infinitive_selecting_verbs",
            pattern_id=f"gerunds_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="The governing verb in this sentence is followed by a to infinitive.",
        ))
    contrast = [
        ("locking", "I remember ___ the door before I left.", ["to lock tomorrow", "lock", "locked"]),
        ("to lock", "Please remember ___ the door when you leave.", ["locking yesterday", "lock", "locked"]),
        ("smoking", "He stopped ___ because of his health.", ["to smoke a cigarette", "smoke", "smoked"]),
        ("to rest", "She stopped ___ for a few minutes before continuing.", ["resting every day", "rest", "rested"]),
        ("meeting", "I will never forget ___ her for the first time.", ["to meet tomorrow", "meet", "met"]),
        ("to send", "Don't forget ___ the attachment tonight.", ["sending yesterday", "send", "sent"]),
        ("working", "He went on ___ despite the noise.", ["to work on a new topic", "work", "worked"]),
        ("to explain", "After describing the problem, she went on ___ the solution.", ["explaining the same point", "explain", "explained"]),
    ]
    for i, (correct, question, wrongs) in enumerate(contrast * 2):
        out.append(_candidate(
            area="gerunds_infinitives", difficulty="hard", rule_id="gerund_infinitive_meaning_contrast",
            pattern_id=f"gerunds_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="The gerund and infinitive can express different meanings after this verb, so choose the form that matches the context.",
        ))
    return out


def _comparatives_superlatives() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    easy = [
        ("faster", "This train is ___ than the local one.", ["fastest", "fast", "more faster"]),
        ("smaller", "My room is ___ than my brother's.", ["smallest", "small", "more smaller"]),
        ("taller", "Anna is ___ than her sister.", ["tallest", "tall", "more taller"]),
        ("colder", "Today is ___ than yesterday.", ["coldest", "cold", "more colder"]),
        ("younger", "My cousin is ___ than I am.", ["youngest", "young", "more younger"]),
        ("longer", "This route is ___ than the other one.", ["longest", "long", "more longer"]),
        ("cheaper", "The bus ticket is ___ than the train ticket.", ["cheapest", "cheap", "more cheaper"]),
        ("brighter", "This lamp is ___ than the old one.", ["brightest", "bright", "more brighter"]),
    ]
    for i, (correct, question, wrongs) in enumerate(easy * 2):
        out.append(_candidate(
            area="comparatives_superlatives", difficulty="easy", rule_id="short_adjective_comparative",
            pattern_id=f"comparison_easy_{i % 4}", variant=i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the comparative form with 'than' when comparing two things.",
        ))
    medium = [
        ("more interesting", "This book is ___ than the first one.", ["interestinger", "most interesting", "interesting"]),
        ("more expensive", "The downtown hotel is ___ than the airport hotel.", ["expensiver", "most expensive", "expensive"]),
        ("more comfortable", "This chair is ___ than that one.", ["comfortabler", "most comfortable", "comfortable"]),
        ("more difficult", "The final exam was ___ than the practice test.", ["difficulter", "most difficult", "difficult"]),
        ("more important", "Accuracy is ___ than speed in this task.", ["importanter", "most important", "important"]),
        ("more reliable", "The new system is ___ than the old one.", ["reliabler", "most reliable", "reliable"]),
        ("more effective", "This treatment is ___ than the previous one.", ["effectiver", "most effective", "effective"]),
        ("more convenient", "Online booking is ___ than calling the office.", ["convenienter", "most convenient", "convenient"]),
    ]
    for i, (correct, question, wrongs) in enumerate(medium * 2):
        out.append(_candidate(
            area="comparatives_superlatives", difficulty="medium", rule_id="long_adjective_comparative",
            pattern_id=f"comparison_medium_{i % 4}", variant=100 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use 'more' plus a longer adjective to form the comparative.",
        ))
    hard = [
        ("earlier", "The sooner we leave, the ___ we will arrive.", ["earliest", "early", "more earliest"]),
        ("better", "The more you practice, the ___ you become.", ["best", "good", "more better"]),
        ("more complicated", "This problem is far ___ than I expected.", ["most complicated", "complicatedest", "complicated"]),
        ("less expensive", "This option is considerably ___ than the premium plan.", ["least expensive", "expensiver", "lesser expensive"]),
        ("worse", "The traffic was even ___ than yesterday.", ["worst", "bad", "more worse"]),
        ("more likely", "With more evidence, that explanation becomes ___ to be correct.", ["most likely", "likelierest", "likely"]),
        ("fewer", "This version contains ___ errors than the previous one.", ["fewest", "less", "more fewer"]),
        ("less", "The revised process takes ___ time than before.", ["least", "fewer", "more less"]),
    ]
    for i, (correct, question, wrongs) in enumerate(hard * 2):
        out.append(_candidate(
            area="comparatives_superlatives", difficulty="hard", rule_id="advanced_comparison_patterns",
            pattern_id=f"comparison_hard_{i % 4}", variant=200 + i,
            question=question, correct=correct, distractors=wrongs,
            explanation="Use the comparative form required by the advanced comparison pattern and the type of noun or adjective.",
        ))
    return out


def _build_catalog() -> Dict[str, List[Dict[str, Any]]]:
    builders = {
        "past_tense": _past_tense,
        "present_tense": _present_tense,
        "future_tense": _future_tense,
        "present_perfect": _present_perfect,
        "articles": _articles,
        "prepositions": _prepositions,
        "subject_verb_agreement": _subject_verb_agreement,
        "verb_forms": _verb_forms,
        "adjectives": _adjectives,
        "adverbs": _adverbs,
        "pronouns": _pronouns,
        "conjunctions": _conjunctions,
        "conditionals": _conditionals,
        "modals": _modals,
        "sentence_structure": _sentence_structure,
        "word_order": _word_order,
        "countable_uncountable_nouns": _countable_uncountable,
        "plurals": _plurals,
        "gerunds_infinitives": _gerunds_infinitives,
        "comparatives_superlatives": _comparatives_superlatives,
    }
    catalog: Dict[str, List[Dict[str, Any]]] = {}
    for area, builder in builders.items():
        seen = set()
        unique: List[Dict[str, Any]] = []
        for item in builder():
            key = (item["difficulty"], normalize_question(item["question"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        catalog[area] = unique
    return catalog


STRUCTURED_PRACTICE_CATALOG = _build_catalog()


def structured_candidates(area: str, difficulty: str) -> List[Dict[str, Any]]:
    normalized_difficulty = str(difficulty).strip().lower()
    if normalized_difficulty == "advanced":
        normalized_difficulty = "hard"
    if normalized_difficulty not in {"easy", "medium", "hard"}:
        normalized_difficulty = "medium"

    bank = STRUCTURED_PRACTICE_CATALOG.get(area, [])
    exact = [item for item in bank if item.get("difficulty") == normalized_difficulty]
    return [dict(item, choices=dict(item["choices"])) for item in (exact or bank)]


def select_structured_candidate(
    *,
    area: str,
    difficulty: str,
    recent_records: Iterable[Dict[str, Any]],
    similarity_threshold: float = 0.86,
) -> Optional[Dict[str, Any]]:
    candidates = structured_candidates(area, difficulty)
    if not candidates:
        return None

    records = [item for item in recent_records if isinstance(item, dict)]
    recent_questions = [str(item.get("question", "")) for item in records if item.get("question")]
    recent_patterns = [str(item.get("pattern_id", "")) for item in records[-6:] if item.get("pattern_id")]
    pattern_counts = Counter(str(item.get("pattern_id", "")) for item in records if item.get("pattern_id"))
    rule_counts = Counter(str(item.get("rule_id", "")) for item in records if item.get("rule_id"))

    fresh = [
        item for item in candidates
        if not is_near_duplicate_question(item["question"], recent_questions, threshold=similarity_threshold)
    ]

    requested = "hard" if difficulty == "advanced" else difficulty
    difficulty_order = {"easy": 0, "medium": 1, "hard": 2}

    if fresh:
        pool = fresh
    else:
        # If the current difficulty has been exhausted recently, widen to the
        # rest of the structured catalog before repeating a near duplicate.
        all_bank = [dict(item, choices=dict(item["choices"])) for item in STRUCTURED_PRACTICE_CATALOG.get(area, [])]
        fresh_any = [
            item for item in all_bank
            if not is_near_duplicate_question(item["question"], recent_questions, threshold=similarity_threshold)
        ]
        if fresh_any:
            pool = fresh_any
        else:
            # Near duplicate filtering is intentionally strict, but an unseen
            # vetted question is still preferable to an exact repeat.
            seen_exact = {normalize_question(value) for value in recent_questions}
            unseen_exact = [
                item for item in all_bank
                if normalize_question(item["question"]) not in seen_exact
            ]
            pool = unseen_exact or candidates

    def rank(item: Dict[str, Any]) -> tuple[int, int, int, int, str]:
        pattern = str(item.get("pattern_id", ""))
        rule = str(item.get("rule_id", ""))
        item_difficulty = str(item.get("difficulty", "medium"))
        distance = abs(
            difficulty_order.get(item_difficulty, 1)
            - difficulty_order.get(requested, 1)
        )
        return (
            distance,
            1 if pattern in recent_patterns[-4:] else 0,
            pattern_counts.get(pattern, 0),
            rule_counts.get(rule, 0),
            str(item.get("candidate_id", "")),
        )

    chosen = min(pool, key=rank)
    result = dict(chosen)
    result["choices"] = dict(chosen["choices"])
    return result


def catalog_stats() -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for area, bank in STRUCTURED_PRACTICE_CATALOG.items():
        stats[area] = {
            difficulty: sum(1 for item in bank if item["difficulty"] == difficulty)
            for difficulty in ("easy", "medium", "hard")
        }
        stats[area]["total"] = len(bank)
    return stats



# ============================================================
# PRACTICE TEACHING FEEDBACK
# ============================================================

RULE_REMINDERS: Dict[str, str] = {
    "simple_past_regular": "Use the simple past for a completed action at a finished past time.",
    "simple_past_irregular": "Irregular verbs use their irregular past form for completed past actions.",
    "past_irregular_context": "Use a past form that matches the completed past-time context.",

    "present_habit_third_person": "In the simple present, he, she, and it normally take a verb ending in -s or -es.",
    "present_habit_plural_subject": "Plural subjects use the base form of the verb in the simple present.",
    "general_truth_present": "Use the simple present for general facts, routines, and truths.",

    "will_prediction": "Use will + base verb for a neutral prediction about the future.",
    "going_to_evidence": "Use be going to when a future prediction is based on present evidence.",
    "future_perfect_deadline": "Use will have + past participle for something completed before a future deadline.",

    "present_result": "Use the present perfect for a past action that has a result or relevance now.",
    "present_perfect_for_duration": "Use the present perfect with for or since when a situation began in the past and continues now.",
    "life_experience_ever": "Use have or has + past participle for life experience up to the present.",

    "indefinite_article_sound": "Choose a or an by the first sound of the next word, not simply by its first letter.",
    "definite_article_conventional": "Use the when the listener can identify the specific or unique thing being referred to.",
    "article_exceptions_zero_and_sound": "Some expressions use no article, and a versus an is determined by pronunciation.",

    "basic_time_place_prepositions": "Time and place expressions use conventional prepositions such as in, on, at, under, and between.",
    "prepositional_collocations": "Many verbs and adjectives form fixed combinations with particular prepositions.",
    "advanced_prepositional_collocations": "Learn the whole verb or adjective pattern together with its preposition.",

    "basic_subject_verb_number": "The finite verb must agree in number with the grammatical subject.",
    "agreement_with_complex_subjects": "Find the head noun of the subject and make the verb agree with that noun.",
    "special_agreement_patterns": "Special subjects such as amounts and neither/or structures follow specific agreement rules.",

    "base_form_after_modal": "A modal verb is followed directly by the base form of the next verb.",
    "auxiliary_required_form": "Auxiliary verbs determine which form of the following verb is required.",
    "complex_nonfinite_and_perfect_forms": "Perfect, passive, and nonfinite constructions require a specific verb form.",

    "ed_ing_adjectives_cause": "Use an -ing adjective for the thing or situation that causes a feeling.",
    "ed_ing_adjectives_feeling": "Use an -ed adjective for the person or group experiencing the feeling.",
    "adjective_after_linking_or_complex_verb": "Linking verbs and some complex verb patterns take an adjective complement.",

    "adverb_modifies_action": "Use an adverb to describe how an action is performed.",
    "adverb_form_and_irregular_adverbs": "Use the correct adverb form when modifying a verb, adjective, or another adverb.",
    "advanced_adverb_collocation": "Choose the adverb whose form and meaning fit the phrase it modifies.",

    "object_pronouns": "Use an object pronoun when the pronoun receives the action of the verb.",
    "object_pronouns_after_prepositions": "After a preposition, use the object form of a personal pronoun.",
    "relative_pronoun_case": "Choose who or whom according to the pronoun's grammatical function inside the relative clause.",

    "basic_clause_conjunctions": "Choose the conjunction that matches the relationship between the two ideas.",
    "subordinating_conjunction_meaning": "A subordinating conjunction should express the intended reason, contrast, time, or condition.",
    "advanced_clause_linkers": "Advanced linkers differ in meaning, so choose the one that precisely matches the relationship between clauses.",

    "zero_and_first_conditionals": "Zero and first conditionals use different tense patterns for facts and real future possibilities.",
    "second_conditional": "A second conditional normally uses a past form in the if-clause and would + base verb in the result clause.",
    "third_conditional": "A third conditional uses had + past participle and would have + past participle for an unreal past situation.",

    "basic_modal_form": "Modal verbs do not take -s and are not followed by infinitival to.",
    "modal_plus_base_verb": "Use the base form directly after a modal such as can, should, must, or might.",
    "modal_perfect_and_semi_modal": "Modal perfect and semi-modal constructions require a specific base or participle form.",

    "dependent_plus_independent_clause": "A dependent clause must be attached to a complete independent clause.",
    "complete_main_clause_after_complex_subject": "A complete sentence needs a finite main verb after its full subject phrase.",
    "logical_subject_of_participial_modifier": "The subject after an opening participial phrase must logically be the person or thing performing that action.",

    "frequency_adverb_position": "Frequency adverbs such as always and usually normally come before the main verb.",
    "embedded_question_statement_order": "Embedded questions use statement word order, not direct-question inversion.",
    "inversion_after_fronted_expression": "Certain fronted negative or limiting expressions require subject-auxiliary inversion.",

    "basic_uncountable_nouns": "Uncountable nouns such as information and furniture normally do not take a plural -s.",
    "quantifiers_by_countability": "Choose quantifiers according to whether the noun is countable or uncountable.",
    "countable_units_for_mass_nouns": "Use a countable unit such as piece of when giving an exact number with an uncountable noun.",

    "irregular_plural_nouns": "Some nouns form their plural irregularly instead of simply adding -s.",
    "invariant_plural_nouns": "Some nouns have the same form in the singular and plural.",
    "advanced_plural_forms": "Some nouns follow special spelling or inherited plural patterns.",

    "gerund_selecting_verbs": "Some verbs are followed by a gerund, such as enjoy doing and avoid doing.",
    "infinitive_selecting_verbs": "Some verbs are followed by a to-infinitive, such as decide to do and hope to do.",
    "gerund_infinitive_meaning_contrast": "With some verbs, a gerund and an infinitive express different meanings.",

    "short_adjective_comparative": "Short adjectives usually form the comparative with -er.",
    "long_adjective_comparative": "Longer adjectives usually form the comparative with more.",
    "advanced_comparison_patterns": "Comparison structures must use the comparative or superlative form required by the sentence pattern.",
}


PREPOSITION_PATTERN_NOTES = [
    ("born ___", "in", "be born in + month/year/place", "Use 'in' with months, years, and larger places in this expression."),
    ("keys are ___ the table", "on", "on + surface", "Use 'on' when something is resting on a surface."),
    ("starts ___ nine", "at", "at + exact clock time", "Use 'at' with a specific clock time."),
    ("class ___ monday", "on", "on + day/date", "Use 'on' with days and dates."),
    ("live ___ canada", "in", "live in + country/city", "Use 'in' with countries and cities after 'live'."),
    ("meet you ___ the station", "at", "at + specific point/place", "Use 'at' for a specific meeting point such as a station."),
    ("hiding ___ the chair", "under", "under + object", "Use 'under' when something is below another object."),
    ("bank is ___ the cafe", "between", "between A and B", "Use 'between' when something is in the middle of two things."),

    ("interested ___", "in", "interested in + noun/gerund", "The adjective 'interested' is followed by 'in'."),
    ("responsible ___", "for", "responsible for + noun/gerund", "The adjective 'responsible' is followed by 'for'."),
    ("afraid ___", "of", "afraid of + noun/gerund", "The adjective 'afraid' is followed by 'of'."),
    ("listen ___", "to", "listen to + noun", "The verb 'listen' is followed by 'to' before its object."),
    ("agree ___", "with", "agree with + person/idea", "Use 'agree with' when agreeing with a person, opinion, or point."),
    ("complained ___", "about", "complain about + thing/problem", "Use 'complain about' for the thing that causes the complaint."),
    ("apologized ___", "for", "apologize for + thing/action", "Use 'apologize for' for the reason or action being apologized for."),
    ("depends ___", "on", "depend on + noun", "The verb 'depend' is followed by 'on'."),

    ("succeeded ___", "in", "succeed in + gerund", "Use 'succeed in doing something'."),
    ("divided ___", "into", "divide into + parts", "Use 'divide into' when something is separated into parts."),
    ("differs ___", "from", "differ from + noun", "Use 'differ from' when comparing something with something different."),
    ("consistent ___", "with", "consistent with + noun", "The adjective 'consistent' is followed by 'with'."),
    ("accused ___", "of", "accuse someone of + gerund/noun", "Use 'accuse someone of doing something'."),
    ("objected ___", "to", "object to + noun/gerund", "Use 'object to doing something'; here 'to' is a preposition."),
    ("warned us ___", "against", "warn someone against + gerund", "Use 'warn someone against doing something' when advising them not to do it."),
    ("prevent her ___", "from", "prevent someone from + gerund", "Use 'prevent someone from doing something'."),
]


def _choice_text_for_sentence(choice_text: str) -> str:
    value = str(choice_text).strip()
    if value.lower() in {"no article", "zero article", "no determiner", "∅"}:
        return ""
    return value


def completed_practice_sentence(question: str, choice_text: str) -> str:
    replacement = _choice_text_for_sentence(choice_text)
    completed = str(question).replace("___", replacement, 1)
    completed = re.sub(r"\s+([,.!?;:])", r"\1", completed)
    completed = re.sub(r"\s{2,}", " ", completed)
    return completed.strip()


def _preposition_feedback(
    question: str,
    selected_text: str,
    correct_text: str,
) -> Optional[Dict[str, str]]:
    normalized = normalize_question(question)

    for marker, expected, pattern, reason in PREPOSITION_PATTERN_NOTES:
        if marker in normalized and correct_text.strip().lower() == expected:
            return {
                "why_wrong": (
                    f"Choosing '{selected_text}' gives “{completed_practice_sentence(question, selected_text)}”. "
                    f"That preposition does not fit this pattern. English uses “{pattern}”."
                ),
                "why_correct": (
                    f"'{correct_text}' gives “{completed_practice_sentence(question, correct_text)}”. "
                    f"{reason}"
                ),
                "rule": pattern,
            }

    return None


def _article_feedback(
    question: str,
    selected_text: str,
    correct_text: str,
    base_explanation: str,
) -> Dict[str, str]:
    after_blank = str(question).split("___", 1)[1].strip() if "___" in str(question) else ""
    next_word_match = re.match(r"([A-Za-z][A-Za-z'-]*)", after_blank)
    next_word = next_word_match.group(1) if next_word_match else "the following word"
    correct_normalized = correct_text.strip().lower()

    if correct_normalized in {"a", "an"}:
        sound_type = "a vowel sound" if correct_normalized == "an" else "a consonant sound"
        why_correct = (
            f"'{correct_text}' is correct because '{next_word}' begins with {sound_type}. "
            "The choice between 'a' and 'an' depends on pronunciation, not only spelling."
        )
        rule = "Use 'a' before a consonant sound and 'an' before a vowel sound."
    elif correct_normalized == "the":
        why_correct = (
            f"'the' gives “{completed_practice_sentence(question, correct_text)}”. "
            f"{base_explanation}"
        )
        rule = RULE_REMINDERS["definite_article_conventional"]
    else:
        why_correct = (
            f"No article gives “{completed_practice_sentence(question, correct_text)}”. "
            f"{base_explanation}"
        )
        rule = RULE_REMINDERS["article_exceptions_zero_and_sound"]

    return {
        "why_wrong": (
            f"Choosing '{selected_text}' gives “{completed_practice_sentence(question, selected_text)}”. "
            f"That does not match the article rule used here. {rule}"
        ),
        "why_correct": why_correct,
        "rule": rule,
    }


def build_answer_feedback(
    *,
    grammar_area: str,
    rule_id: Optional[str],
    question: str,
    choices: Dict[str, str],
    student_answer: str,
    correct_answer: str,
    base_explanation: str,
) -> Dict[str, str]:
    """Build learner-facing feedback that teaches the mistake and the rule."""
    selected_text = str(choices[student_answer]).strip()
    correct_text = str(choices[correct_answer]).strip()
    correct_sentence = completed_practice_sentence(question, correct_text)
    selected_sentence = completed_practice_sentence(question, selected_text)
    normalized_rule = str(rule_id or "").strip()

    rule = RULE_REMINDERS.get(
        normalized_rule,
        "Choose the form that fits the grammar pattern used in the whole sentence.",
    )

    special: Optional[Dict[str, str]] = None
    if grammar_area == "prepositions":
        special = _preposition_feedback(question, selected_text, correct_text)
    elif grammar_area == "articles":
        special = _article_feedback(
            question,
            selected_text,
            correct_text,
            base_explanation,
        )

    if special:
        why_correct = special["why_correct"]
        why_wrong = special["why_wrong"]
        rule = special["rule"]
    else:
        why_correct = (
            f"'{correct_text}' gives “{correct_sentence}”. "
            f"{base_explanation.strip()} "
            f"That matches the rule: {rule}"
        )
        why_wrong = (
            f"Choosing '{selected_text}' gives “{selected_sentence}”. "
            f"That choice does not fit the grammar pattern required here. "
            f"{rule}"
        )

    if student_answer == correct_answer:
        why_wrong = ""

    return {
        "selected_choice_text": selected_text,
        "correct_choice_text": correct_text,
        "selected_sentence": selected_sentence,
        "correct_sentence": correct_sentence,
        "why_selected_wrong": why_wrong,
        "why_correct": why_correct,
        "rule_to_remember": rule,
    }
