from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.run_pipeline import run_pipeline
from src.utils.paths import resolve_path


if __name__ == "__main__":
    run_pipeline(
        resolve_path("configs"),
        model_name="gmm_reduced",
        pipeline_name="full_reduced",
        from_stage="evaluate",
        to_stage="diagnostics",
    )
