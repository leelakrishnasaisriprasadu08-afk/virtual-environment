import tempfile
import unittest
from pathlib import Path

from virtual_environment.environment import LearningEnvironment
from virtual_environment.memory import StudentMemory, validate_profile_id
from virtual_environment.models import TemplateStudentModel
from virtual_environment.scenes import SCENES


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.memory = StudentMemory(Path(self.temp_dir.name) / "student.sqlite3")
        self.environment = LearningEnvironment(self.memory, TemplateStudentModel())

    def tearDown(self):
        self.memory.close()
        self.temp_dir.cleanup()

    def test_template_student_passes_each_builtin_scene(self):
        for scene in SCENES:
            with self.subTest(scene=scene.scene_id):
                result = self.environment.practice("student", scene.scene_id)
                self.assertTrue(result.passed)
                self.assertGreaterEqual(result.score, 2 / 3)

    def test_failed_response_is_stored_with_actionable_feedback(self):
        result = SCENES[0].evaluate("I don't know.")
        score, passed, feedback = result
        self.assertEqual(score, 0)
        self.assertFalse(passed)
        self.assertIn("Try adding", feedback)

    def test_criteria_do_not_match_accidental_substrings(self):
        score, passed, _ = SCENES[0].evaluate("This is a classmate.")
        self.assertEqual(score, 0)
        self.assertFalse(passed)

    def test_memory_is_profile_scoped_and_can_be_deleted(self):
        self.memory.set_preference("alpha", "style", "examples")
        self.memory.set_preference("beta", "style", "short")
        self.environment.practice("alpha", "introductions")

        self.assertEqual(self.memory.preferences("alpha"), [("style", "examples")])
        self.assertEqual(self.memory.preferences("beta"), [("style", "short")])
        self.assertEqual(len(self.memory.progress("beta")), 0)
        self.assertEqual(self.memory.delete_profile("alpha"), (1, 1))
        self.assertEqual(self.memory.preferences("alpha"), [])
        self.assertEqual(self.memory.progress("alpha"), [])

    def test_memory_is_not_passed_to_model_without_explicit_opt_in(self):
        class RecordingModel:
            name = "recording"

            def __init__(self):
                self.arguments = None

            def respond(self, scene, successful_lessons, preferences):
                self.arguments = successful_lessons, preferences
                return "Hello! What's your name? Nice to meet you."

        recording = RecordingModel()
        environment = LearningEnvironment(self.memory, recording)
        self.memory.set_preference("student", "style", "examples")
        environment.practice("student", "introductions")
        self.assertEqual(recording.arguments, ([], []))

        environment.practice("student", "introductions", share_memory_with_model=True)
        self.assertEqual(recording.arguments[1], [("style", "examples")])

    def test_invalid_profile_ids_are_rejected(self):
        for profile_id in ("../outside", "has space", "", "x" * 51):
            with self.subTest(profile_id=profile_id):
                with self.assertRaises(ValueError):
                    validate_profile_id(profile_id)


if __name__ == "__main__":
    unittest.main()
