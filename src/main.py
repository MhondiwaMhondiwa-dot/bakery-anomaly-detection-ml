import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.eda_restart_guide_aligned import run

if __name__ == "__main__":
    # Default: prepare processed data, then run analyses
    run(prepare=True)
