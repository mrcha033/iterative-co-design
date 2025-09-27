#!/usr/bin/env python3
"""Audit presence of remediation artifacts promised in the gap plan."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Requirement:
    """A filesystem requirement expressed as a path or glob pattern."""

    pattern: str

    def evaluate(self) -> dict[str, object]:
        pattern = self.pattern
        is_glob = any(ch in pattern for ch in "*?[]")
        if is_glob:
            matches = [p for p in ROOT.glob(pattern) if p.exists()]
            exists = bool(matches)
            sample = matches[0].as_posix() if matches else None
        else:
            path = ROOT / pattern
            exists = path.exists()
            sample = path.as_posix() if exists else None
        return {
            "pattern": pattern,
            "exists": exists,
            "sample": sample,
        }


@dataclass(frozen=True)
class GapItem:
    key: str
    description: str
    requirements: Sequence[Requirement]
    require_all: bool = True

    def evaluate(self) -> dict[str, object]:
        results: List[dict[str, object]] = [req.evaluate() for req in self.requirements]
        satisfied = all(r["exists"] for r in results) if self.require_all else any(
            r["exists"] for r in results
        )
        return {
            "key": self.key,
            "description": self.description,
            "satisfied": satisfied,
            "requirements": results,
        }


@dataclass(frozen=True)
class GapSection:
    name: str
    items: Sequence[GapItem]

    def evaluate(self) -> dict[str, object]:
        evaluated = [item.evaluate() for item in self.items]
        satisfied = all(entry["satisfied"] for entry in evaluated)
        return {
            "section": self.name,
            "satisfied": satisfied,
            "items": evaluated,
        }


SECTIONS: Sequence[GapSection] = (
    GapSection(
        name="Structured Sparsity (HDS)",
        items=(
            GapItem(
                key="hds_layer_audit",
                description="Layer eligibility audit report(s)",
                requirements=(Requirement("reports/hds/*.json"),),
                require_all=False,
            ),
            GapItem(
                key="hds_mask_state",
                description="Mask checkpoint schema and test",
                requirements=(
                    Requirement("docs/schema/hds_mask_state.json"),
                    Requirement("tests/hds/test_mask_checkpoint.py"),
                ),
            ),
            GapItem(
                key="hds_defaults",
                description="Default HDS config and annealing guardrail test",
                requirements=(
                    Requirement("configs/hds_default.yaml"),
                    Requirement("tests/hds/test_annealing_guardrail.py"),
                ),
            ),
            GapItem(
                key="hds_export",
                description="Sparse export utilities and golden artifact",
                requirements=(
                    Requirement("icd/runtime/export_sparse.py"),
                    Requirement("tests/data/hds/sparse_export.pt"),
                ),
            ),
        ),
    ),
    GapSection(
        name="Iterative Correlation & Clustering (IASP)",
        items=(
            GapItem(
                key="iasp_manifest_schema",
                description="Correlation capture manifest schema",
                requirements=(Requirement("docs/schema/iasp_capture_manifest.json"),),
            ),
            GapItem(
                key="iasp_capture_tool",
                description="Correlation capture CLI helper",
                requirements=(Requirement("scripts/iasp_capture_dump.py"),),
            ),
            GapItem(
                key="iasp_transfer_resilience",
                description="Transfer resilience test and log sample",
                requirements=(
                    Requirement("tests/iasp/test_transfer_resilience.py"),
                    Requirement("logs/iasp_transfer.log"),
                ),
            ),
            GapItem(
                key="iasp_defaults",
                description="IASP configuration defaults and documentation",
                requirements=(
                    Requirement("configs/iasp_defaults.yaml"),
                    Requirement("docs/IASP_Config.md"),
                ),
            ),
            GapItem(
                key="iasp_clustering_guardrail",
                description="Clustering guardrail tests",
                requirements=(Requirement("tests/iasp/test_clustering_switch.py"),),
            ),
            GapItem(
                key="iasp_perm_cache",
                description="Permutation cache schema and tests",
                requirements=(
                    Requirement("docs/schema/perm_cache_v1.json"),
                    Requirement("tests/iasp/test_perm_cache.py"),
                ),
            ),
        ),
    ),
    GapSection(
        name="Acceptance Gates & Rollback",
        items=(
            GapItem(
                key="measurement_manifest",
                description="Measurement manifest schema",
                requirements=(Requirement("docs/schema/measurement_manifest.json"),),
            ),
            GapItem(
                key="rollback_flow",
                description="Rollback flow documentation and tests",
                requirements=(
                    Requirement("docs/rollback_flow.mmd"),
                    Requirement("tests/runtime/test_rollback.py"),
                ),
            ),
            GapItem(
                key="rca_templates",
                description="RCA templates and examples",
                requirements=(
                    Requirement("docs/templates/rca_template.md"),
                    Requirement("docs/examples/rca_sample.md"),
                ),
            ),
            GapItem(
                key="completeness_metric",
                description="Completeness scoring guide and script",
                requirements=(
                    Requirement("docs/Completeness_Scoring.md"),
                    Requirement("scripts/compute_completeness.py"),
                ),
            ),
        ),
    ),
    GapSection(
        name="Measurement Observability",
        items=(
            GapItem(
                key="nsight_stub",
                description="Nsight orchestration stubs",
                requirements=(
                    Requirement("scripts/nsight_stub.py"),
                    Requirement("tests/measure/test_nsight_stub.py"),
                ),
            ),
            GapItem(
                key="nvml_logger",
                description="NVML logging implementation and tests",
                requirements=(
                    Requirement("icd/runtime/nvml_logger.py"),
                    Requirement("tests/measure/test_nvml_logger.py"),
                ),
            ),
            GapItem(
                key="env_fingerprint",
                description="Environment fingerprint schema and privacy review",
                requirements=(
                    Requirement("docs/schema/env_fingerprint.json"),
                    Requirement("docs/Env_Fingerprint_Privacy.md"),
                ),
            ),
            GapItem(
                key="regression_baseline_guide",
                description="Regression baseline ownership guide",
                requirements=(Requirement("docs/Regression_Baseline_Guide.md"),),
            ),
        ),
    ),
    GapSection(
        name="IR Pass Integration",
        items=(
            GapItem(
                key="stablehlo_capability_detection",
                description="StableHLO capability detection",
                requirements=(Requirement("icd/runtime/apply_pi.py"),),
            ),
            GapItem(
                key="metrics_bridge",
                description="StableHLO metrics bridge implementation and tests",
                requirements=(
                    Requirement("icd/measure/metrics_bridge.py"),
                    Requirement("tests/ir/test_metrics_bridge.py"),
                ),
            ),
            GapItem(
                key="stablehlo_build",
                description="StableHLO build guide and workflow",
                requirements=(
                    Requirement("docs/StableHLO_Build.md"),
                    Requirement(".github/workflows/stablehlo-build.yml"),
                ),
            ),
        ),
    ),
    GapSection(
        name="QA & Reproducibility",
        items=(
            GapItem(
                key="ci_matrix",
                description="CI matrix documentation",
                requirements=(Requirement("docs/CI_Matrix.md"),),
            ),
            GapItem(
                key="verify_env_lock",
                description="Environment locking script",
                requirements=(Requirement("scripts/verify_env_lock.py"),),
            ),
            GapItem(
                key="iterative_checklist",
                description="Iterative co-design readiness checklist",
                requirements=(Requirement("docs/Iterative_CoDesign_Checklist.md"),),
            ),
        ),
    ),
    GapSection(
        name="Timeline & Ownership Risks",
        items=(
            GapItem(
                key="resource_plan",
                description="Resource plan documentation",
                requirements=(Requirement("docs/Resource_Plan.md"),),
            ),
            GapItem(
                key="dependency_dag",
                description="Dependency DAG diagram",
                requirements=(Requirement("docs/Dependency_DAG.mmd"),),
            ),
            GapItem(
                key="milestone_definitions",
                description="Milestone definitions",
                requirements=(Requirement("docs/Milestone_Definitions.md"),),
            ),
        ),
    ),
    GapSection(
        name="Documentation & Communication",
        items=(
            GapItem(
                key="doc_update_backlog",
                description="Documentation update backlog",
                requirements=(Requirement("docs/Doc_Update_Backlog.md"),),
            ),
            GapItem(
                key="change_announcement_playbook",
                description="Change announcement process",
                requirements=(Requirement("docs/Change_Announcement_Playbook.md"),),
            ),
            GapItem(
                key="external_reference_policy",
                description="External reference policy and PR template linkage",
                requirements=(Requirement("docs/External_Reference_Policy.md"),),
            ),
        ),
    ),
)


def evaluate_sections(sections: Iterable[GapSection]) -> list[dict[str, object]]:
    return [section.evaluate() for section in sections]


def render_text_report(results: Sequence[dict[str, object]]) -> str:
    lines: list[str] = []
    for section in results:
        status = "OK" if section["satisfied"] else "MISSING"
        lines.append(f"# {section['section']} — {status}")
        for item in section["items"]:
            icon = "✔" if item["satisfied"] else "✘"
            lines.append(f"- {icon} {item['description']}")
            for req in item["requirements"]:
                mark = "•" if req["exists"] else "○"
                sample = f" ({req['sample']})" if req["sample"] else ""
                lines.append(f"    {mark} {req['pattern']}{sample}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    results = evaluate_sections(SECTIONS)
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text_report(results))


if __name__ == "__main__":
    main()
