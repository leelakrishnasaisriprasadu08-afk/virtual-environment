import unittest

from virtual_environment.models import OpenAICompatibleModel, build_model
from virtual_environment.scenes import SCENES_BY_ID


class ModelTests(unittest.TestCase):
    def test_template_model_is_local_and_matches_scene_rubric(self):
        from virtual_environment.models import TemplateStudentModel

        model = TemplateStudentModel()
        response = model.respond(SCENES_BY_ID["introductions"], [], [])
        score, passed, _ = SCENES_BY_ID["introductions"].evaluate(response)
        self.assertTrue(passed)
        self.assertGreaterEqual(score, 2 / 3)

    def test_remote_http_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleModel("secret", "model", "http://example.com/v1")

    def test_local_http_endpoint_is_allowed(self):
        model = OpenAICompatibleModel("local", "model", "http://localhost:8000/v1")
        self.assertEqual(model.base_url, "http://localhost:8000/v1")

    def test_model_url_cannot_embed_credentials(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleModel("secret", "model", "https://user:password@example.com/v1")

    def test_api_model_requires_environment_key(self):
        from unittest.mock import patch

        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "MODEL_API_KEY"):
                build_model("api")


if __name__ == "__main__":
    unittest.main()
