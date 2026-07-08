"""Move legacy result folders under one central ``results/experiments`` tree."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


MOVE_RULES = [
    ("results/metrics", "results/experiments/baseline_100e/metrics"),
    ("results/metrics_v2", "results/experiments/initial_simmixup_v2/metrics"),
    ("results/metrics_v3", "results/experiments/initial_simcutmix_v3/metrics"),
    ("results/experiments_v2", "results/experiments/simmixup_ablation_v2"),
    ("results/experiments_v3", "results/experiments/anchor_gated_v3"),
    ("results/final_stage", "results/experiments/final_stage_v1"),
    ("results/checkpoints", "results/experiments/shared/checkpoints"),
    ("results/neighbors", "results/experiments/shared/neighbors"),
    ("results/anchor_scores", "results/experiments/shared/anchor_scores"),
    ("results/figures", "results/experiments/shared/figures"),
]

PREFIX_REWRITES = [
    ("results/experiments_v2", "results/experiments/simmixup_ablation_v2"),
    ("results/experiments_v3", "results/experiments/anchor_gated_v3"),
    ("results/final_stage", "results/experiments/final_stage_v1"),
    ("results/metrics_v2", "results/experiments/initial_simmixup_v2/metrics"),
    ("results/metrics_v3", "results/experiments/initial_simcutmix_v3/metrics"),
    ("results/metrics", "results/experiments/baseline_100e/metrics"),
    ("results/checkpoints", "results/experiments/shared/checkpoints"),
    ("results/neighbors", "results/experiments/shared/neighbors"),
    ("results/anchor_scores", "results/experiments/shared/anchor_scores"),
    ("results/figures", "results/experiments/shared/figures"),
]

OUTPUT_ROOT_BY_COLLECTION = {
    "baseline_100e": "results/experiments/baseline_100e",
    "initial_simmixup_v2": "results/experiments/initial_simmixup_v2",
    "initial_simcutmix_v3": "results/experiments/initial_simcutmix_v3",
}


def normalize(value: str) -> str:
    return value.replace("\\", "/")


def rewrite_path_string(value: str) -> str:
    normalized = normalize(value)
    for old, new in PREFIX_REWRITES:
        if normalized == old:
            return new
        if normalized.startswith(old + "/"):
            return new + normalized[len(old) :]
    return normalized


def collection_for_summary(path: Path) -> str | None:
    parts = path.parts
    try:
        idx = parts.index("experiments")
    except ValueError:
        return None
    if len(parts) > idx + 1:
        return parts[idx + 1]
    return None


def update_json_values(payload: Any, summary_path: Path) -> Any:
    if isinstance(payload, dict):
        updated: dict[str, Any] = {}
        collection = collection_for_summary(summary_path)
        for key, value in payload.items():
            if isinstance(value, str):
                if key == "output_root" and normalize(value) == "results" and collection in OUTPUT_ROOT_BY_COLLECTION:
                    updated[key] = OUTPUT_ROOT_BY_COLLECTION[collection]
                elif normalize(value).startswith("results"):
                    updated[key] = rewrite_path_string(value)
                else:
                    updated[key] = value
            else:
                updated[key] = update_json_values(value, summary_path)
        return updated
    if isinstance(payload, list):
        return [update_json_values(item, summary_path) for item in payload]
    return payload


def update_text(text: str) -> str:
    updated = text.replace("\\", "/")
    for old, new in PREFIX_REWRITES:
        updated = updated.replace(old, new)
    return updated


def ensure_inside(path: Path, root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to operate outside {resolved_root}: {resolved_path}")


def move_legacy_dirs(repo_root: Path) -> None:
    experiments_root = repo_root / "results" / "experiments"
    experiments_root.mkdir(parents=True, exist_ok=True)
    ensure_inside(experiments_root, repo_root / "results")

    for source_rel, target_rel in MOVE_RULES:
        source = repo_root / source_rel
        target = repo_root / target_rel
        ensure_inside(source, repo_root / "results")
        ensure_inside(target, experiments_root)

        if not source.exists():
            if target.exists():
                print(f"skip existing target: {target_rel}")
                continue
            print(f"skip missing source: {source_rel}")
            continue
        if target.exists():
            raise FileExistsError(f"Target already exists, not merging: {target_rel}")

        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"move {source_rel} -> {target_rel}")
        shutil.move(str(source), str(target))


def rewrite_json_files(repo_root: Path) -> None:
    for path in sorted((repo_root / "results" / "experiments").rglob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = update_json_values(payload, path)
        if updated != payload:
            path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
            print(f"rewrite json: {path.relative_to(repo_root).as_posix()}")


def rewrite_markdown_files(repo_root: Path) -> None:
    candidates = [
        repo_root / "notes" / "experiment_results_summary_v1.md",
        repo_root / "notes" / "experiments_v2_results.md",
        repo_root / "baseline_results.md",
        repo_root / "docs" / "project_status.md",
        repo_root / "results" / "experiments" / "README.md",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = update_text(text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"rewrite text: {path.relative_to(repo_root).as_posix()}")


def main() -> None:
    repo_root = Path.cwd()
    move_legacy_dirs(repo_root)
    rewrite_json_files(repo_root)
    rewrite_markdown_files(repo_root)


if __name__ == "__main__":
    main()
