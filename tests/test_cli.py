import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliWorkflowTests(unittest.TestCase):
    def test_practice_progress_and_profile_deletion_persist_across_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "student.sqlite3"

            practice = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "virtual_environment.cli",
                    "--model",
                    "demo",
                    "--profile",
                    "smoke-test",
                    "--data",
                    str(database),
                ],
                input="1\n6\n",
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                check=True,
            )
            self.assertIn("Score: 100% | PASS", practice.stdout)

            progress = subprocess.run(
                [sys.executable, "-m", "virtual_environment.cli", "--profile", "smoke-test", "--data", str(database)],
                input="4\n6\n",
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                check=True,
            )
            self.assertIn("introductions: passed, 1 attempt(s)", progress.stdout)

            deletion = subprocess.run(
                [sys.executable, "-m", "virtual_environment.cli", "--profile", "smoke-test", "--data", str(database)],
                input="5\nDELETE\n4\n6\n",
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                env=os.environ.copy(),
                check=True,
            )
            self.assertIn("Deleted 1 practice record(s)", deletion.stdout)
            self.assertIn("No practice recorded yet.", deletion.stdout)


if __name__ == "__main__":
    unittest.main()
