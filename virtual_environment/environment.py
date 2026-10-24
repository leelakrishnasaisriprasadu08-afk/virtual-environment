"""Environment loop joining curated scenes, model responses, evaluation, and memory."""

from dataclasses import dataclass

from .memory import StudentMemory, validate_profile_id
from .models import StudentModel
from .scenes import SCENES_BY_ID, Scene


@dataclass(frozen=True)
class PracticeResult:
    scene: Scene
    response: str
    score: float
    passed: bool
    feedback: str


class LearningEnvironment:
    def __init__(self, memory: StudentMemory, student: StudentModel):
        self.memory = memory
        self.student = student

    def practice(
        self,
        profile_id: str,
        scene_id: str,
        share_memory_with_model: bool = False,
    ) -> PracticeResult:
        validate_profile_id(profile_id)
        try:
            scene = SCENES_BY_ID[scene_id]
        except KeyError as error:
            raise ValueError(f"Unknown scene: {scene_id}") from error

        lessons = self.memory.successful_lessons(profile_id) if share_memory_with_model else []
        preferences = self.memory.preferences(profile_id) if share_memory_with_model else []
        response = self.student.respond(scene, lessons, preferences)
        score, passed, feedback = scene.evaluate(response)
        self.memory.save_experience(profile_id, scene_id, response, score, passed, feedback)
        return PracticeResult(scene, response, score, passed, feedback)
