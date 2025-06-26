from datetime import datetime, timedelta
from utils.cleanup import cleanup_old_runs

def test_cleanup_old_runs(tmp_path):
    # Create some dummy files and directories
    now = datetime.now()
    old_dir = tmp_path / (now - timedelta(days=40)).strftime("%Y-%m")
    new_dir = tmp_path / now.strftime("%Y-%m")
    old_dir.mkdir(exist_ok=True)
    new_dir.mkdir(exist_ok=True)
    (old_dir / "run1").mkdir()
    (new_dir / "run2").mkdir()

    cleanup_old_runs([str(tmp_path)], 30)

    assert not (old_dir / "run1").exists()
    assert (new_dir / "run2").exists()

def test_cleanup_dry_run(tmp_path):
    # Create some dummy files and directories
    now = datetime.now()
    old_dir = tmp_path / (now - timedelta(days=40)).strftime("%Y-%m")
    new_dir = tmp_path / now.strftime("%Y-%m")
    old_dir.mkdir(exist_ok=True)
    new_dir.mkdir(exist_ok=True)
    (old_dir / "run1").mkdir()
    (new_dir / "run2").mkdir()

    cleanup_old_runs([str(tmp_path)], 30, dry_run=True)

    assert (old_dir / "run1").exists()
    assert (new_dir / "run2").exists()
