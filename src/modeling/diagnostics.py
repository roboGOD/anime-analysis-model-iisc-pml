from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.io import write_json


def run(model_metadata_path: Path, reports_dir: Path, logger: logging.Logger, overwrite: bool = False) -> dict[str, str]:
    stem = model_metadata_path.stem.replace("_model_metadata", "")
    output_path = reports_dir / f"{stem}_diagnostics.json"
    if output_path.exists() and not overwrite:
        logger.info("Skipping diagnostics; output already exists at %s", output_path)
        return {"diagnostics": str(output_path)}

    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    diagnostics = {
        "model_name": metadata["model_name"],
        "converged": metadata.get("converged"),
        "n_iter": metadata.get("n_iter"),
        "lower_bound": metadata.get("lower_bound"),
        "training_params": metadata.get("training_params", {}),
    }
    if "weights" in metadata:
        diagnostics["component_weight_summary"] = {
            "min_weight": min(metadata["weights"]),
            "max_weight": max(metadata["weights"]),
        }
    write_json(diagnostics, output_path)
    logger.info("Saved diagnostics to %s", output_path)
    return {"diagnostics": str(output_path)}
