import unittest
from pathlib import Path
from datetime import datetime, timedelta
import shutil
from utils.cleanup import cleanup_old_runs

class TestCleanup(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path("./test_cleanup_dir")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_cleanup_old_runs(self):
        # Create some dummy files and directories
        now = datetime.now()
        old_dir = self.test_dir / (now - timedelta(days=40)).strftime("%Y-%m")
        new_dir = self.test_dir / now.strftime("%Y-%m")
        old_dir.mkdir(exist_ok=True)
        new_dir.mkdir(exist_ok=True)
        (old_dir / "run1").mkdir()
        (new_dir / "run2").mkdir()

        cleanup_old_runs([str(self.test_dir)], 30)

        self.assertFalse((old_dir / "run1").exists())
        self.assertTrue((new_dir / "run2").exists())

    def test_cleanup_dry_run(self):
        # Create some dummy files and directories
        now = datetime.now()
        old_dir = self.test_dir / (now - timedelta(days=40)).strftime("%Y-%m")
        new_dir = self.test_dir / now.strftime("%Y-%m")
        old_dir.mkdir(exist_ok=True)
        new_dir.mkdir(exist_ok=True)
        (old_dir / "run1").mkdir()
        (new_dir / "run2").mkdir()

        cleanup_old_runs([str(self.test_dir)], 30, dry_run=True)

        self.assertTrue((old_dir / "run1").exists())
        self.assertTrue((new_dir / "run2").exists())

if __name__ == '__main__':
    unittest.main()