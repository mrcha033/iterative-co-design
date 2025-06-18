"""Utilities for cleaning up old experiment outputs."""

import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def cleanup_old_runs(
    base_dirs: List[str],
    max_age_days: int,
    dry_run: bool = False,
    exclude_patterns: Optional[List[str]] = None
) -> None:
    """Clean up old experiment outputs.
    
    Args:
        base_dirs: List of base directories to clean (e.g. ["outputs", "multirun"])
        max_age_days: Maximum age in days before deletion
        dry_run: If True, only print what would be deleted without actually deleting
        exclude_patterns: List of patterns to exclude from deletion
    """
    exclude_patterns = exclude_patterns or []
    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue
            
        # Walk through year-month directories
        for year_month in base_path.iterdir():
            if not year_month.is_dir():
                continue
                
            try:
                # Parse year-month directory name (format: YYYY-MM)
                year, month = map(int, year_month.name.split("-"))
                dir_date = datetime(year, month, 1)
                
                # Skip if directory is newer than cutoff
                if dir_date > cutoff_date:
                    continue
                    
                # Check each run directory
                for run_dir in year_month.iterdir():
                    if not run_dir.is_dir():
                        continue
                        
                    # Skip if matches exclude pattern
                    if any(pattern in str(run_dir) for pattern in exclude_patterns):
                        continue
                        
                    if dry_run:
                        logger.info(f"Would delete: {run_dir}")
                    else:
                        try:
                            shutil.rmtree(run_dir)
                            logger.info(f"Deleted: {run_dir}")
                        except Exception as e:
                            logger.error(f"Failed to delete {run_dir}: {e}")
                            
                # Try to remove empty month directory
                if not any(year_month.iterdir()):
                    year_month.rmdir()
                    
            except (ValueError, OSError) as e:
                logger.warning(f"Skipping invalid directory {year_month}: {e}")
                continue 
