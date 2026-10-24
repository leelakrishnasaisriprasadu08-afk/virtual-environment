"""Curated scenes and simple, inspectable evaluation rules."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Criterion:
    label: str
    alternatives: tuple[str, ...]

    def matches(self, response: str) -> bool:
        normalized = re.sub(r"[^\w]+", " ", response.casefold()).strip()
        padded_response = f" {normalized} "
        return any(
            (term == "?" and "?" in response)
            or f" {re.sub(r'[^\w]+', ' ', term.casefold()).strip()} " in padded_response
            for term in self.alternatives
        )


@dataclass(frozen=True)
class Scene:
    scene_id: str
    topic: str
    situation: str
    example: str
    task: str
    criteria: tuple[Criterion, ...]
    preference_candidate: tuple[str, str] | None = None

    def evaluate(self, response: str) -> tuple[float, bool, str]:
        matched = [criterion for criterion in self.criteria if criterion.matches(response)]
        score = len(matched) / len(self.criteria)
        passed = score >= 2 / 3
        met = [criterion.label for criterion in matched]
        missing = [criterion.label for criterion in self.criteria if criterion not in matched]

        if passed:
            feedback = "Goal met. Demonstrated: " + ", ".join(met) + "."
        else:
            feedback = "Not yet. Demonstrated: " + (", ".join(met) or "none")
            feedback += ". Try adding: " + ", ".join(missing) + "."
        return score, passed, feedback


SCENES = (
    Scene(
        scene_id="introductions",
        topic="Introductions",
        situation="You are meeting a classmate for the first time.",
        example="A: Hi, I'm Jordan. What should I call you?\nB: Hello, I'm Lee. Nice to meet you.",
        task="Greet the classmate, ask their name, and respond warmly.",
        criteria=(
            Criterion("a greeting", ("hello", "hi", "good morning", "good afternoon")),
            Criterion("a question about their name", ("what is your name", "what's your name", "your name", "what should i call")),
            Criterion("a friendly response", ("nice to meet", "pleased to meet", "glad to meet")),
        ),
    ),
    Scene(
        scene_id="asking-for-help",
        topic="Asking for help",
        situation="You are stuck on a class assignment and a classmate is nearby.",
        example="A: Could you help me understand this question?\nB: Sure. Which part is confusing?",
        task="Ask politely for specific help and briefly explain what you are trying to do.",
        criteria=(
            Criterion("a polite request", ("please", "could you", "can you", "would you")),
            Criterion("a request for help", ("help me", "help with", "assist me", "show me")),
            Criterion("brief context", ("because", "i'm trying", "i am trying", "i need", "this task")),
        ),
    ),
    Scene(
        scene_id="making-a-plan",
        topic="Making a plan",
        situation="You and a classmate have a group assignment due next week.",
        example="A: Let's list the tasks first, then choose who will do each one.\nB: Good idea. What should we start with?",
        task="Suggest a first step, describe what should happen next, and invite the classmate's input.",
        criteria=(
            Criterion("a suggestion", ("let's", "we could", "we should", "i suggest", "how about")),
            Criterion("an ordered step", ("first", "then", "next", "start by", "begin by")),
            Criterion("an invitation for input", ("what do you think", "does that work", "would that work", "your thoughts", "?")),
        ),
    ),
    Scene(
        scene_id="noticing-preferences",
        topic="Noticing preferences",
        situation="During a practice conversation, Sam says: 'I understand new ideas best when I can see a concrete example. Long explanations without examples are hard for me.'",
        example="A: Examples help me understand new ideas.\nB: Would you like me to use examples when I explain things?",
        task="Reflect the preference you observed and ask Sam to confirm it.",
        criteria=(
            Criterion("the preference for examples", ("example", "examples", "demonstration")),
            Criterion("recognizing what helps Sam", ("you prefer", "you learn", "helps you understand", "easier for you")),
            Criterion("a confirmation question", ("would you", "do you prefer", "is that right", "should i", "?")),
        ),
        preference_candidate=("explanation_style", "prefers examples"),
    ),
)

SCENES_BY_ID = {scene.scene_id: scene for scene in SCENES}
