"""Experiment tracking for evaluation runs."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


@dataclass
class Experiment:
    id: str
    name: str
    timestamp: str
    prompt_version: str = "unknown"
    model: str = "unknown"
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_score: float = 0.0
    results: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def log_experiment(experiment: Experiment) -> str:
    """
    Save experiment results to JSON file.

    Returns:
        Path to the saved experiment file.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = EXPERIMENTS_DIR / f"{experiment.id}.json"

    with open(file_path, "w") as f:
        json.dump(asdict(experiment), f, indent=2, default=str)

    logger.info("Experiment logged", id=experiment.id, file=str(file_path))
    return str(file_path)


def list_experiments() -> list[dict]:
    """List all saved experiments."""
    experiments = []
    if not EXPERIMENTS_DIR.exists():
        return experiments

    for json_file in sorted(EXPERIMENTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(json_file) as f:
                data = json.load(f)
            experiments.append({
                "id": data.get("id"),
                "name": data.get("name"),
                "timestamp": data.get("timestamp"),
                "total_tests": data.get("total_tests"),
                "passed_tests": data.get("passed_tests"),
                "average_score": data.get("average_score"),
            })
        except Exception as e:
            logger.warning("Failed to load experiment", file=str(json_file), error=str(e))

    return experiments


def compare_experiments(id1: str, id2: str) -> Optional[dict]:
    """Compare two experiment runs."""
    exp1_path = EXPERIMENTS_DIR / f"{id1}.json"
    exp2_path = EXPERIMENTS_DIR / f"{id2}.json"

    if not exp1_path.exists() or not exp2_path.exists():
        return None

    with open(exp1_path) as f:
        exp1 = json.load(f)
    with open(exp2_path) as f:
        exp2 = json.load(f)

    return {
        "experiment_1": {
            "id": exp1["id"],
            "name": exp1["name"],
            "average_score": exp1["average_score"],
            "passed": exp1["passed_tests"],
            "total": exp1["total_tests"],
        },
        "experiment_2": {
            "id": exp2["id"],
            "name": exp2["name"],
            "average_score": exp2["average_score"],
            "passed": exp2["passed_tests"],
            "total": exp2["total_tests"],
        },
        "score_delta": exp2["average_score"] - exp1["average_score"],
        "pass_rate_delta": (
            (exp2["passed_tests"] / max(exp2["total_tests"], 1))
            - (exp1["passed_tests"] / max(exp1["total_tests"], 1))
        ),
    }
