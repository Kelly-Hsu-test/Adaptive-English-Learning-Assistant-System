from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "learner_data.json"

VALID_GRAMMAR_AREAS = [
    "past_tense",
    "present_tense",
    "future_tense",
    "present_perfect",
    "articles",
    "prepositions",
    "subject_verb_agreement",
    "verb_forms",
    "adjectives",
    "adverbs",
    "pronouns",
    "conjunctions",
    "conditionals",
    "modals",
    "sentence_structure",
    "word_order",
    "countable_uncountable_nouns",
    "plurals",
    "gerunds_infinitives",
    "comparatives_superlatives",
]

PENDING_EXERCISE_TTL_HOURS = 24
MAX_PENDING_EXERCISES = 50
MAX_PRACTICE_HISTORY = 300
MAX_CEFR_HISTORY = 100

_FILE_LOCK = threading.RLock()


GRAMMAR_ALIASES = {
    "adjective": "adjectives",
    "adverb": "adverbs",
    "pronoun": "pronouns",
    "article": "articles",
    "preposition": "prepositions",
    "verb_form": "verb_forms",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_grammar_area(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    area = value.strip().lower()
    area = GRAMMAR_ALIASES.get(area, area)

    if area in VALID_GRAMMAR_AREAS:
        return area

    return None


def _default_skill() -> Dict[str, Any]:
    return {
        "writing_errors": 0,
        "writing_samples_with_errors": 0,
        "practice_attempts": 0,
        "practice_correct": 0,
        "practice_incorrect": 0,
        "practice_accuracy": 0.0,
        "recent_accuracy": None,
        "recent_attempts": 0,
        "recent_correct": 0,
        "consecutive": {
            "type": None,
            "count": 0,
        },
        "weakness_score": 0.0,
    }


def _empty_profile() -> Dict[str, Any]:
    return {
        "grammar_errors": {},
        "vocabulary": {},
        "cefr_history": [],
        "current_cefr": None,
        "practice": {
            "total_attempts": 0,
            "total_correct": 0,
            "total_incorrect": 0,
            "overall_accuracy": 0.0,
        },
        "skills": {},
        "recommendation": None,
        "practice_history": [],
        "pending_exercises": {},
    }


DEFAULT_PROFILE = _empty_profile()


def _normalize_profile(data: Any) -> Dict[str, Any]:
    profile = _empty_profile()

    if isinstance(data, dict):
        for key in profile:
            if key in data:
                profile[key] = data[key]

    if not isinstance(profile["grammar_errors"], dict):
        profile["grammar_errors"] = {}

    if not isinstance(profile["vocabulary"], dict):
        profile["vocabulary"] = {}

    if not isinstance(profile["cefr_history"], list):
        profile["cefr_history"] = []

    if not isinstance(profile["practice_history"], list):
        profile["practice_history"] = []

    if not isinstance(profile["skills"], dict):
        profile["skills"] = {}

    if not isinstance(profile["pending_exercises"], dict):
        profile["pending_exercises"] = {}

    if not isinstance(profile["practice"], dict):
        profile["practice"] = {}

    practice = profile["practice"]
    practice["total_attempts"] = _safe_int(practice.get("total_attempts"))
    practice["total_correct"] = _safe_int(practice.get("total_correct"))
    practice["total_incorrect"] = _safe_int(practice.get("total_incorrect"))

    attempts = practice["total_attempts"]
    correct = practice["total_correct"]
    practice["overall_accuracy"] = (
        round(correct / attempts * 100, 2)
        if attempts > 0
        else 0.0
    )

    normalized_errors: Dict[str, int] = {}
    for raw_area, raw_count in profile["grammar_errors"].items():
        area = _normalize_grammar_area(raw_area)
        if area:
            normalized_errors[area] = normalized_errors.get(area, 0) + max(
                0,
                _safe_int(raw_count),
            )
    profile["grammar_errors"] = normalized_errors

    for area in VALID_GRAMMAR_AREAS:
        existing = profile["skills"].get(area, {})
        skill = _default_skill()

        if isinstance(existing, dict):
            skill.update(existing)

        if not isinstance(skill.get("consecutive"), dict):
            skill["consecutive"] = {
                "type": None,
                "count": 0,
            }

        skill["writing_errors"] = max(
            0,
            _safe_int(skill.get("writing_errors")),
        )
        skill["writing_samples_with_errors"] = max(
            0,
            _safe_int(skill.get("writing_samples_with_errors")),
        )
        skill["practice_attempts"] = max(
            0,
            _safe_int(skill.get("practice_attempts")),
        )
        skill["practice_correct"] = max(
            0,
            _safe_int(skill.get("practice_correct")),
        )
        skill["practice_incorrect"] = max(
            0,
            _safe_int(skill.get("practice_incorrect")),
        )

        practice_attempts = skill["practice_attempts"]
        skill["practice_accuracy"] = (
            round(skill["practice_correct"] / practice_attempts * 100, 2)
            if practice_attempts > 0
            else 0.0
        )
        skill["weakness_score"] = max(
            0.0,
            min(100.0, _safe_float(skill.get("weakness_score"))),
        )

        profile["skills"][area] = skill

    profile["practice_history"] = [
        item
        for item in profile["practice_history"]
        if isinstance(item, dict)
    ][-MAX_PRACTICE_HISTORY:]

    profile["cefr_history"] = [
        str(level).strip().upper()
        for level in profile["cefr_history"]
        if str(level).strip().upper() in {"A1", "A2", "B1", "B2", "C1", "C2"}
    ][-MAX_CEFR_HISTORY:]

    if profile["current_cefr"] not in {"A1", "A2", "B1", "B2", "C1", "C2"}:
        profile["current_cefr"] = (
            profile["cefr_history"][-1]
            if profile["cefr_history"]
            else None
        )

    normalized_pending: Dict[str, Dict[str, Any]] = {}
    for exercise_id, exercise in profile["pending_exercises"].items():
        if not isinstance(exercise, dict):
            continue

        try:
            normalized_id = str(uuid.UUID(str(exercise_id)))
        except (ValueError, TypeError, AttributeError):
            continue

        area = _normalize_grammar_area(exercise.get("grammar_area"))
        choices = exercise.get("choices")
        correct_answer = str(exercise.get("correct_answer", "")).strip().upper()
        difficulty = str(exercise.get("difficulty", "medium")).strip().lower()
        question = exercise.get("question")
        explanation = exercise.get("explanation")
        created_at = exercise.get("created_at")
        rule_id = str(exercise.get("rule_id", "")).strip() or None
        pattern_id = str(exercise.get("pattern_id", "")).strip() or None
        source = str(exercise.get("source", "")).strip() or None
        practice_signature = str(exercise.get("practice_signature", "")).strip() or None

        if not area:
            continue
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(explanation, str) or not explanation.strip():
            continue
        if not isinstance(choices, dict) or set(choices) != {"A", "B", "C", "D"}:
            continue
        if not all(isinstance(value, str) and value.strip() for value in choices.values()):
            continue
        if correct_answer not in {"A", "B", "C", "D"}:
            continue
        if difficulty not in {"easy", "medium", "hard", "advanced"}:
            difficulty = "medium"
        if not isinstance(created_at, str) or not created_at.strip():
            created_at = _utc_now_iso()

        normalized_pending[normalized_id] = {
            "exercise_id": normalized_id,
            "grammar_area": area,
            "question": question.strip(),
            "choices": {key: str(value).strip() for key, value in choices.items()},
            "correct_answer": correct_answer,
            "explanation": explanation.strip(),
            "difficulty": difficulty,
            "created_at": created_at,
            "rule_id": rule_id,
            "pattern_id": pattern_id,
            "source": source,
            "practice_signature": practice_signature,
        }

    profile["pending_exercises"] = normalized_pending

    for area in VALID_GRAMMAR_AREAS:
        recent = [
            item
            for item in profile["practice_history"]
            if (
                item.get("grammar_area") == area
                and isinstance(item.get("correct"), bool)
            )
        ][-10:]

        skill = profile["skills"][area]
        skill["recent_attempts"] = len(recent)
        skill["recent_correct"] = sum(
            1 for item in recent if item.get("correct") is True
        )
        skill["recent_accuracy"] = (
            round(skill["recent_correct"] / len(recent) * 100, 2)
            if recent
            else None
        )

    return profile


def extract_grammar_errors(
    analysis: Any,
    unique: bool = True,
) -> List[str]:
    if not isinstance(analysis, dict):
        return []

    grammar = analysis.get("grammar", [])
    if not isinstance(grammar, list):
        return []

    errors: List[str] = []
    seen = set()

    for item in grammar:
        if not isinstance(item, dict):
            continue

        area = _normalize_grammar_area(item.get("type"))
        if not area:
            continue

        if unique:
            if area in seen:
                continue
            seen.add(area)

        errors.append(area)

    return errors


class LearnerProfile:
    def __init__(self, data_file: Optional[Path] = None):
        self.data_file = Path(data_file) if data_file else DATA_FILE
        self.profile = self._load()
        self._cleanup_pending_exercises(save=False)

    def _load(self) -> Dict[str, Any]:
        if not self.data_file.exists():
            return _normalize_profile({})

        with _FILE_LOCK:
            try:
                with self.data_file.open("r", encoding="utf-8") as file:
                    data = json.load(file)
                return _normalize_profile(data)
            except (OSError, json.JSONDecodeError, TypeError):
                return _normalize_profile({})

    def _save(self) -> None:
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.data_file.with_suffix(self.data_file.suffix + ".tmp")

        with _FILE_LOCK:
            with temporary_file.open("w", encoding="utf-8") as file:
                json.dump(
                    self.profile,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
            temporary_file.replace(self.data_file)

    def _cleanup_pending_exercises(self, save: bool = True) -> None:
        pending = self.profile.get("pending_exercises", {})
        if not isinstance(pending, dict):
            self.profile["pending_exercises"] = {}
            if save:
                self._save()
            return

        cutoff = datetime.now(timezone.utc) - timedelta(hours=PENDING_EXERCISE_TTL_HOURS)
        kept: List[tuple[str, Dict[str, Any], datetime]] = []

        for exercise_id, exercise in pending.items():
            if not isinstance(exercise, dict):
                continue

            created_raw = exercise.get("created_at")
            try:
                created_at = datetime.fromisoformat(str(created_raw))
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                created_at = datetime.now(timezone.utc)

            if created_at >= cutoff:
                kept.append((exercise_id, exercise, created_at))

        kept.sort(key=lambda item: item[2])
        kept = kept[-MAX_PENDING_EXERCISES:]
        cleaned = {exercise_id: exercise for exercise_id, exercise, _ in kept}

        changed = cleaned != pending
        self.profile["pending_exercises"] = cleaned

        if changed and save:
            self._save()

    def get_profile(self) -> Dict[str, Any]:
        self.profile = _normalize_profile(self.profile)
        self._cleanup_pending_exercises(save=False)
        return self.profile

    def get_public_profile(self) -> Dict[str, Any]:
        self.profile = _normalize_profile(self.profile)
        self._update_recommendation()
        self._cleanup_pending_exercises(save=False)

        skills: Dict[str, Dict[str, Any]] = {}
        for area, skill in self.profile["skills"].items():
            skills[area] = {
                "writing_errors": skill["writing_errors"],
                "writing_samples_with_errors": skill["writing_samples_with_errors"],
                "practice_attempts": skill["practice_attempts"],
                "practice_correct": skill["practice_correct"],
                "practice_incorrect": skill["practice_incorrect"],
                "practice_accuracy": skill["practice_accuracy"],
                "recent_accuracy": skill["recent_accuracy"],
                "recent_attempts": skill["recent_attempts"],
                "consecutive": dict(skill["consecutive"]),
                "weakness_score": skill["weakness_score"],
                "recommended_difficulty": self._choose_difficulty(skill),
            }

        recent_practice = []
        for item in self.profile["practice_history"]:
            if not isinstance(item.get("correct"), bool):
                continue

            recent_practice.append(
                {
                    "exercise_id": item.get("exercise_id"),
                    "grammar_area": item.get("grammar_area"),
                    "question": item.get("question"),
                    "student_answer": item.get("student_answer"),
                    "correct": item.get("correct"),
                    "difficulty": item.get("difficulty"),
                    "answered_at": item.get("answered_at"),
                    "rule_id": item.get("rule_id"),
                    "pattern_id": item.get("pattern_id"),
                    "source": item.get("source"),
                }
            )

        recent_answered = recent_practice[-10:]
        recent_correct = sum(
            1 for item in recent_answered if item.get("correct") is True
        )
        recent_accuracy = (
            round(recent_correct / len(recent_answered) * 100, 2)
            if recent_answered
            else None
        )

        practice_summary = dict(self.profile["practice"])
        practice_summary["recent_accuracy"] = recent_accuracy
        practice_summary["recent_attempts"] = len(recent_answered)

        return {
            "grammar_errors": dict(self.profile["grammar_errors"]),
            "current_cefr": self.profile["current_cefr"],
            "cefr_history": list(self.profile["cefr_history"][-20:]),
            "practice": practice_summary,
            "skills": skills,
            "recommendation": (
                dict(self.profile["recommendation"])
                if isinstance(self.profile["recommendation"], dict)
                else None
            ),
            "recent_practice": recent_practice[-20:],
        }

    def record_writing_analysis(self, error_types: Iterable[str]) -> None:
        normalized = [
            area
            for area in (_normalize_grammar_area(value) for value in error_types)
            if area
        ]

        if not normalized:
            return

        counts = Counter(normalized)

        for area, count in counts.items():
            self.profile["grammar_errors"][area] = (
                _safe_int(self.profile["grammar_errors"].get(area, 0)) + count
            )

            skill = self.profile["skills"][area]
            skill["writing_errors"] += count
            skill["writing_samples_with_errors"] += 1
            self._recalculate_weakness(area)

        self._update_recommendation()
        self._save()

    def record_grammar_error(self, error_type: str) -> None:
        self.record_writing_analysis([error_type])

    def record_cefr_level(self, level: str) -> None:
        normalized = str(level).strip().upper()
        if normalized not in {"A1", "A2", "B1", "B2", "C1", "C2"}:
            return

        self.profile["cefr_history"].append(normalized)
        self.profile["cefr_history"] = self.profile["cefr_history"][-MAX_CEFR_HISTORY:]
        self.profile["current_cefr"] = normalized
        self._save()

    def record_practice_question(
        self,
        grammar_area: str,
        question: str,
        difficulty: str,
        exercise_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        pattern_id: Optional[str] = None,
        source: Optional[str] = None,
        practice_signature: Optional[str] = None,
    ) -> None:
        area = _normalize_grammar_area(grammar_area)
        if not area:
            return

        self.profile["practice_history"].append(
            {
                "exercise_id": exercise_id,
                "grammar_area": area,
                "question": str(question),
                "difficulty": str(difficulty),
                "generated": True,
                "generated_at": _utc_now_iso(),
            }
        )
        self.profile["practice_history"] = self.profile["practice_history"][-MAX_PRACTICE_HISTORY:]
        self._save()

    def create_practice_exercise(
        self,
        grammar_area: str,
        question: str,
        choices: Dict[str, str],
        correct_answer: str,
        explanation: str,
        difficulty: str,
        rule_id: Optional[str] = None,
        pattern_id: Optional[str] = None,
        source: Optional[str] = None,
        practice_signature: Optional[str] = None,
    ) -> str:
        area = _normalize_grammar_area(grammar_area)
        if not area:
            raise ValueError("Invalid grammar area.")

        normalized_choices = {
            str(key).strip().upper(): str(value).strip()
            for key, value in choices.items()
        }
        if set(normalized_choices) != {"A", "B", "C", "D"}:
            raise ValueError("Choices must contain exactly A, B, C, and D.")
        if not all(normalized_choices.values()):
            raise ValueError("Choices cannot be empty.")

        normalized_answer = str(correct_answer).strip().upper()
        if normalized_answer not in normalized_choices:
            raise ValueError("Correct answer must be A, B, C, or D.")

        normalized_difficulty = str(difficulty).strip().lower()
        if normalized_difficulty not in {"easy", "medium", "hard", "advanced"}:
            normalized_difficulty = "medium"

        normalized_question = str(question).strip()
        normalized_explanation = str(explanation).strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")
        if not normalized_explanation:
            raise ValueError("Explanation cannot be empty.")

        normalized_rule_id = str(rule_id).strip() if rule_id else None
        normalized_pattern_id = str(pattern_id).strip() if pattern_id else None
        normalized_source = str(source).strip() if source else None
        normalized_signature = (
            str(practice_signature).strip() if practice_signature else None
        )

        exercise_id = str(uuid.uuid4())
        self._cleanup_pending_exercises(save=False)

        self.profile["pending_exercises"][exercise_id] = {
            "exercise_id": exercise_id,
            "grammar_area": area,
            "question": normalized_question,
            "choices": normalized_choices,
            "correct_answer": normalized_answer,
            "explanation": normalized_explanation,
            "difficulty": normalized_difficulty,
            "created_at": _utc_now_iso(),
            "rule_id": normalized_rule_id,
            "pattern_id": normalized_pattern_id,
            "source": normalized_source,
            "practice_signature": normalized_signature,
        }

        self.profile["practice_history"].append(
            {
                "exercise_id": exercise_id,
                "grammar_area": area,
                "question": normalized_question,
                "difficulty": normalized_difficulty,
                "generated": True,
                "generated_at": _utc_now_iso(),
                "rule_id": normalized_rule_id,
                "pattern_id": normalized_pattern_id,
                "source": normalized_source,
                "practice_signature": normalized_signature,
            }
        )
        self.profile["practice_history"] = self.profile["practice_history"][-MAX_PRACTICE_HISTORY:]
        self._cleanup_pending_exercises(save=False)
        self._save()
        return exercise_id

    def get_pending_exercise(self, exercise_id: str) -> Optional[Dict[str, Any]]:
        self._cleanup_pending_exercises(save=True)

        try:
            normalized_id = str(uuid.UUID(str(exercise_id)))
        except (ValueError, TypeError, AttributeError):
            return None

        exercise = self.profile["pending_exercises"].get(normalized_id)
        if not isinstance(exercise, dict):
            return None

        return dict(exercise)

    def submit_practice_answer(
        self,
        exercise_id: str,
        student_answer: str,
    ) -> Optional[Dict[str, Any]]:
        self._cleanup_pending_exercises(save=False)

        try:
            normalized_id = str(uuid.UUID(str(exercise_id)))
        except (ValueError, TypeError, AttributeError):
            return None

        exercise = self.profile["pending_exercises"].get(normalized_id)
        if not isinstance(exercise, dict):
            return None

        answer = str(student_answer).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError("Student answer must be A, B, C, or D.")

        correct_answer = exercise["correct_answer"]
        is_correct = answer == correct_answer

        self._record_practice_result(
            grammar_area=exercise["grammar_area"],
            correct=is_correct,
            question=exercise["question"],
            student_answer=answer,
            correct_answer=correct_answer,
            difficulty=exercise["difficulty"],
            exercise_id=normalized_id,
            rule_id=exercise.get("rule_id"),
            pattern_id=exercise.get("pattern_id"),
            source=exercise.get("source"),
            practice_signature=exercise.get("practice_signature"),
        )

        del self.profile["pending_exercises"][normalized_id]
        self._save()

        return {
            "exercise_id": normalized_id,
            "correct": is_correct,
            "grammar_area": exercise["grammar_area"],
            "question": exercise["question"],
            "choices": dict(exercise["choices"]),
            "student_answer": answer,
            "correct_answer": correct_answer,
            "correct_choice_text": exercise["choices"][correct_answer],
            "explanation": exercise["explanation"],
            "difficulty": exercise["difficulty"],
            "rule_id": exercise.get("rule_id"),
            "pattern_id": exercise.get("pattern_id"),
            "source": exercise.get("source"),
        }

    def record_practice_result(
        self,
        grammar_area: str,
        correct: bool,
        question: str,
        student_answer: str,
        correct_answer: str,
        difficulty: str,
        exercise_id: Optional[str] = None,
    ) -> None:
        self._record_practice_result(
            grammar_area=grammar_area,
            correct=correct,
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            difficulty=difficulty,
            exercise_id=exercise_id,
            rule_id=rule_id,
            pattern_id=pattern_id,
            source=source,
            practice_signature=practice_signature,
        )
        self._save()

    def _record_practice_result(
        self,
        grammar_area: str,
        correct: bool,
        question: str,
        student_answer: str,
        correct_answer: str,
        difficulty: str,
        exercise_id: Optional[str],
        rule_id: Optional[str] = None,
        pattern_id: Optional[str] = None,
        source: Optional[str] = None,
        practice_signature: Optional[str] = None,
    ) -> None:
        area = _normalize_grammar_area(grammar_area)
        if not area:
            raise ValueError("Invalid grammar area.")

        practice = self.profile["practice"]
        practice["total_attempts"] += 1

        if correct:
            practice["total_correct"] += 1
        else:
            practice["total_incorrect"] += 1

        practice["overall_accuracy"] = round(
            practice["total_correct"] / practice["total_attempts"] * 100,
            2,
        )

        skill = self.profile["skills"][area]
        skill["practice_attempts"] += 1

        if correct:
            skill["practice_correct"] += 1
        else:
            skill["practice_incorrect"] += 1

        self.profile["practice_history"].append(
            {
                "exercise_id": exercise_id,
                "grammar_area": area,
                "question": str(question),
                "student_answer": str(student_answer).strip().upper(),
                "correct_answer": str(correct_answer).strip().upper(),
                "correct": bool(correct),
                "difficulty": str(difficulty).strip().lower(),
                "generated": False,
                "answered_at": _utc_now_iso(),
                "rule_id": str(rule_id).strip() if rule_id else None,
                "pattern_id": str(pattern_id).strip() if pattern_id else None,
                "source": str(source).strip() if source else None,
                "practice_signature": (
                    str(practice_signature).strip() if practice_signature else None
                ),
            }
        )
        self.profile["practice_history"] = self.profile["practice_history"][-MAX_PRACTICE_HISTORY:]

        skill["consecutive"] = {
            "type": "correct" if correct else "incorrect",
            "count": self._get_consecutive_count(area, correct),
        }

        self._recalculate_weakness(area)
        self._update_recommendation()

    def _get_consecutive_count(self, grammar_area: str, correct: bool) -> int:
        count = 0

        for item in reversed(self.profile["practice_history"]):
            if (
                not isinstance(item, dict)
                or item.get("grammar_area") != grammar_area
                or not isinstance(item.get("correct"), bool)
            ):
                continue

            if bool(item["correct"]) != bool(correct):
                break

            count += 1

        return count

    def _recent_results(
        self,
        grammar_area: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        results = [
            item
            for item in self.profile["practice_history"]
            if (
                isinstance(item, dict)
                and item.get("grammar_area") == grammar_area
                and isinstance(item.get("correct"), bool)
            )
        ]
        return results[-limit:]

    def _recalculate_weakness(self, area: str) -> None:
        skill = self.profile["skills"][area]
        writing_errors = skill["writing_errors"]
        writing_samples = skill["writing_samples_with_errors"]
        attempts = skill["practice_attempts"]

        skill["practice_accuracy"] = (
            round(skill["practice_correct"] / attempts * 100, 2)
            if attempts > 0
            else 0.0
        )

        recent = self._recent_results(area, 10)
        skill["recent_attempts"] = len(recent)
        skill["recent_correct"] = sum(
            1 for item in recent if item.get("correct") is True
        )
        skill["recent_accuracy"] = (
            round(skill["recent_correct"] / len(recent) * 100, 2)
            if recent
            else None
        )

        score = 0.0

        score += min(writing_samples * 12.0, 36.0)
        score += min(writing_errors * 3.0, 24.0)

        recent_accuracy = skill["recent_accuracy"]
        if recent_accuracy is not None:
            score += (100.0 - recent_accuracy) * 0.40
        elif writing_errors > 0:
            score += 8.0

        consecutive = skill["consecutive"]
        if consecutive.get("type") == "incorrect":
            score += min(_safe_int(consecutive.get("count")) * 5.0, 20.0)

        if attempts >= 5 and skill["practice_accuracy"] >= 85:
            score -= 10.0

        if recent_accuracy is not None and skill["recent_attempts"] >= 5:
            if recent_accuracy >= 90:
                score -= 15.0
            elif recent_accuracy >= 80:
                score -= 8.0

        skill["weakness_score"] = round(
            max(0.0, min(100.0, score)),
            2,
        )

    def _update_recommendation(self) -> None:
        for area in VALID_GRAMMAR_AREAS:
            self._recalculate_weakness(area)

        candidates = []
        for area in VALID_GRAMMAR_AREAS:
            skill = self.profile["skills"][area]
            if skill["writing_errors"] > 0 or skill["practice_attempts"] > 0:
                candidates.append((area, skill))

        if not candidates:
            self.profile["recommendation"] = None
            return

        area, skill = max(
            candidates,
            key=lambda item: item[1]["weakness_score"],
        )

        attempts = skill["practice_attempts"]
        recent_accuracy = skill["recent_accuracy"]
        consecutive = skill["consecutive"]

        if attempts == 0:
            recommendation_type = "initial_practice"
        elif (
            consecutive.get("type") == "incorrect"
            and _safe_int(consecutive.get("count")) >= 2
        ):
            recommendation_type = "needs_review"
        elif (
            recent_accuracy is not None
            and recent_accuracy >= 85
            and attempts >= 3
        ):
            recommendation_type = "progression"
        else:
            recommendation_type = "continued_practice"

        self.profile["recommendation"] = {
            "grammar_area": area,
            "attempts": attempts,
            "accuracy": skill["practice_accuracy"],
            "recent_accuracy": recent_accuracy,
            "consecutive": dict(consecutive),
            "weakness_score": skill["weakness_score"],
            "writing_errors": skill["writing_errors"],
            "writing_samples_with_errors": skill["writing_samples_with_errors"],
            "recommendation": recommendation_type,
            "difficulty": self._choose_difficulty(skill),
        }

    def _choose_difficulty(self, skill: Dict[str, Any]) -> str:
        attempts = skill["practice_attempts"]
        recent_accuracy = skill["recent_accuracy"]

        if attempts < 3:
            return "medium"

        if recent_accuracy is not None and recent_accuracy < 60:
            return "easy"

        if (
            recent_accuracy is not None
            and recent_accuracy >= 90
            and attempts >= 10
        ):
            return "advanced"

        if (
            recent_accuracy is not None
            and recent_accuracy >= 80
            and attempts >= 6
        ):
            return "hard"

        return "medium"

    def get_learning_recommendation(self) -> Optional[Dict[str, Any]]:
        self._update_recommendation()
        self._save()
        recommendation = self.profile["recommendation"]
        return dict(recommendation) if isinstance(recommendation, dict) else None

    def get_practice_target(self, grammar_area: str) -> Dict[str, Any]:
        """Return adaptive stats/difficulty for a manually selected grammar area."""
        area = _normalize_grammar_area(grammar_area)
        if not area:
            raise ValueError("Invalid grammar area.")

        skill = self.profile["skills"][area]
        return {
            "grammar_area": area,
            "attempts": skill["practice_attempts"],
            "accuracy": skill["practice_accuracy"],
            "recent_accuracy": skill["recent_accuracy"],
            "difficulty": self._choose_difficulty(skill),
            "recommendation": "manual_practice",
        }

    def get_weakest_area(self) -> Optional[Dict[str, Any]]:
        self._update_recommendation()

        candidates = []
        for area in VALID_GRAMMAR_AREAS:
            skill = self.profile["skills"][area]
            if skill["writing_errors"] > 0 or skill["practice_attempts"] > 0:
                candidates.append((area, skill))

        if not candidates:
            return None

        area, skill = max(
            candidates,
            key=lambda item: item[1]["weakness_score"],
        )

        return {
            "area": area,
            **dict(skill),
        }

    def get_recent_practice_records(
        self,
        grammar_area: str,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        area = _normalize_grammar_area(grammar_area)
        if not area:
            return []

        records: List[Dict[str, Any]] = []
        seen_exercise_ids = set()

        for item in reversed(self.profile["practice_history"]):
            if not isinstance(item, dict) or item.get("grammar_area") != area:
                continue

            question = item.get("question")
            if not question:
                continue

            exercise_id = item.get("exercise_id")
            # A generated entry and its later answered entry share an ID.
            # Keep only the newest one so recency metadata is not double counted.
            if exercise_id and exercise_id in seen_exercise_ids:
                continue
            if exercise_id:
                seen_exercise_ids.add(exercise_id)

            records.append(
                {
                    "exercise_id": exercise_id,
                    "grammar_area": area,
                    "question": str(question),
                    "difficulty": item.get("difficulty"),
                    "rule_id": item.get("rule_id"),
                    "pattern_id": item.get("pattern_id"),
                    "source": item.get("source"),
                    "practice_signature": item.get("practice_signature"),
                    "correct": item.get("correct"),
                }
            )

            if len(records) >= max(1, int(limit)):
                break

        return list(reversed(records))

    def get_recent_practice_questions(
        self,
        grammar_area: str,
        limit: int = 10,
    ) -> List[str]:
        area = _normalize_grammar_area(grammar_area)
        if not area:
            return []

        questions: List[str] = []

        for item in reversed(self.profile["practice_history"]):
            if not isinstance(item, dict):
                continue
            if item.get("grammar_area") != area:
                continue

            question = item.get("question")
            if not question:
                continue

            normalized_question = str(question)
            if normalized_question not in questions:
                questions.append(normalized_question)

            if len(questions) >= limit:
                break

        return list(reversed(questions))

    def get_practice_stats(self) -> Dict[str, Any]:
        return dict(self.profile["practice"])

    def reset(self) -> None:
        self.profile = _normalize_profile({})

        with _FILE_LOCK:
            if self.data_file.exists():
                self.data_file.unlink()

        self._save()
