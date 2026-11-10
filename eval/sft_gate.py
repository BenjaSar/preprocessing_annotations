"""Apply the D8 gate: did fine-tuning beat the baseline, or beat the noise?

The pipeline decodes with sampling (VLMConfig.do_sample defaults true,
temperature 1.0), so two runs of the same code on the same pages do not give
the same numbers. A fine-tune that scores a point above one baseline run has
not necessarily done anything. The gate therefore compares the candidate
against the SPREAD of several baseline runs that differ only in VLM_SEED.

D8, as agreed 2026-10-27:
  - label recall above the baseline spread (strictly above the best baseline,
    plus --noise-floor; one baseline has no spread of its own, so supply the
    floor there or the check passes on any positive delta)
  - label precision no more than `tolerance` below the baseline median
  - category accuracy no more than `tolerance` below the baseline median
  - no document loses recall against the worst baseline for that document

Reads reports written by eval/room_gt_eval.py; it does not re-score. Run that
first for every baseline and for the candidate.

Usage:
    python eval/sft_gate.py --baseline a.json --baseline b.json \\
        --candidate adapter.json
"""

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

from report_io import emit_report

from preprocessing_annotations.orchestration.mlflow_tracking import (
    add_mlflow_arguments,
    log_artifact,
    log_metrics,
    log_params,
    mlflow_config_from_args,
    non_fatal,
    tracked_run,
)

logger = logging.getLogger(__name__)

RECALL = "label_recall"
PRECISION = "label_precision"
LOCATED = "gt_rooms_located"
MLFLOW_STAGE = "sft_gate"


def _overall(report: Dict[str, Any], metric: str) -> float:
    return float(report["localization"][metric])


def _category_accuracy(report: Dict[str, Any]) -> float:
    return float(report["category"]["accuracy"])


def _document_recall(report: Dict[str, Any]) -> Dict[str, float]:
    return {
        document: float(score[RECALL])
        for document, score in (report.get("per_document") or {}).items()
    }


def _document_located(report: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """GT rooms located per document -- the count behind each recall rate.

    None where a report does not carry the count: the room-count floor then
    cannot be applied to that document, and the caller falls back to the
    strict rate comparison rather than waving the document through.
    """
    return {
        document: (None if score.get(LOCATED) is None else int(score[LOCATED]))
        for document, score in (report.get("per_document") or {}).items()
    }


def _spread(values: Sequence[float]) -> Dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def apply_gate(
    baselines: Sequence[Dict[str, Any]], candidate: Dict[str, Any], tolerance: float,
    min_rooms_lost: int = 2, noise_floor: float = 0.0,
) -> Dict[str, Any]:
    """Score the candidate against the baseline spread. Pure; returns the verdict.

    noise_floor: how far above the best baseline the candidate's recall must
    land. The gate's whole premise is that a win smaller than run-to-run noise
    is not a win, and it reads that noise off the baseline spread -- but ONE
    baseline has a spread of zero, so `> max` then passes on any positive
    delta, however small. Measured case 2026-11-07: sft_cap200_v1 beat
    adapter_real_v1 by .0020 in a one-baseline gate and passed, while the
    five-seed spread measured that same week was .0140 -- the "win" was a
    seventh of the noise. Pass the spread measured elsewhere as the floor when
    comparing against a single run. Default 0.0 keeps the published behaviour;
    `spread_measurable` in the verdict says whether the baselines could supply
    a floor on their own.

    min_rooms_lost: a document counts as regressed only when it falls below
    every baseline AND loses at least this many GT rooms. Measured on the
    three baseline runs, which differ only by VLM_SEED: 8 of 10 documents
    swing by >=1 located room between them, and
    bp_1_masonic_heights_elementary_school_electrical_pages swings exactly one
    (100, 100, 99). So a one-room difference is inside seed noise, and the
    rate-only rule flagged it as a regression wherever three samples happened
    to agree -- bradley_fair (28 GT rooms, all baselines located 3) failed the
    gate on a single room. Default 2 = the smallest loss not yet observed as
    noise. Raising it hides real small-document losses; 1 restores the old,
    strict rule.
    """
    if not baselines:
        raise ValueError("the gate needs at least one baseline report")

    recalls = _spread([_overall(b, RECALL) for b in baselines])
    precisions = _spread([_overall(b, PRECISION) for b in baselines])
    categories = _spread([_category_accuracy(b) for b in baselines])

    candidate_recall = _overall(candidate, RECALL)
    candidate_precision = _overall(candidate, PRECISION)
    candidate_category = _category_accuracy(candidate)

    # Worst baseline per document: a document only "loses" if the candidate
    # falls below every baseline run, not merely below the luckiest one.
    worst_per_document: Dict[str, float] = {}
    for baseline in baselines:
        for document, recall in _document_recall(baseline).items():
            worst_per_document[document] = min(
                worst_per_document.get(document, recall), recall
            )
    worst_located_per_document: Dict[str, Optional[int]] = {}
    for baseline in baselines:
        for document, located in _document_located(baseline).items():
            previous = worst_located_per_document.get(document, located)
            worst_located_per_document[document] = (
                None if previous is None or located is None else min(previous, located)
            )
    candidate_per_document = _document_recall(candidate)
    candidate_located = _document_located(candidate)
    regressed = {}
    for document, worst in worst_per_document.items():
        if document not in candidate_per_document:
            continue
        if candidate_per_document[document] >= worst:
            continue
        baseline_rooms = worst_located_per_document.get(document)
        candidate_rooms = candidate_located.get(document)
        rooms_lost = (
            None if baseline_rooms is None or candidate_rooms is None
            else baseline_rooms - candidate_rooms
        )
        # No counts: fall back to the strict rate rule rather than excuse it.
        if rooms_lost is not None and rooms_lost < min_rooms_lost:
            continue
        regressed[document] = {
            "candidate": round(candidate_per_document[document], 4),
            "worst_baseline": round(worst, 4),
            "rooms_lost": rooms_lost,
        }
    missing_documents = sorted(set(worst_per_document) - set(candidate_per_document))

    # Comparability: every run in a verdict should come from the same pipeline
    # code. Measured 2026-10-29/30 -- baselines, control and adapter ran hours
    # apart while other work edited the pipeline, and nothing flagged it. An
    # "unknown" fingerprint (runs predating the field) cannot be checked, so it
    # is reported but does not by itself fail the check.
    fingerprints = {
        report.get("run_code_fingerprint", "unknown")
        for report in list(baselines) + [candidate]
    }
    known = {f for f in fingerprints if f != "unknown"}

    checks = {
        "recall_above_baseline_spread": candidate_recall > recalls["max"] + noise_floor,
        "precision_within_tolerance": candidate_precision >= precisions["median"] - tolerance,
        "category_within_tolerance": candidate_category >= categories["median"] - tolerance,
        "no_document_lost_recall": not regressed and not missing_documents,
        "one_pipeline_version": len(known) <= 1,
    }
    return {
        "tolerance": tolerance,
        "min_rooms_lost": min_rooms_lost,
        "baselines": len(baselines),
        "noise_floor": noise_floor,
        # A single baseline cannot measure its own run-to-run noise, so a
        # verdict built on one says so rather than reporting range 0 as if
        # that meant the candidate cleared something.
        "spread_measurable": len(baselines) >= 2,
        "recall_margin_over_best_baseline": round(candidate_recall - recalls["max"], 4),
        "baseline_spread": {
            "label_recall": recalls,
            "label_precision": precisions,
            "category_accuracy": categories,
        },
        "candidate": {
            "label_recall": candidate_recall,
            "label_precision": candidate_precision,
            "category_accuracy": candidate_category,
            "run_dir": candidate.get("run_dir"),
        },
        "deltas_vs_baseline_median": {
            "label_recall": round(candidate_recall - recalls["median"], 4),
            "label_precision": round(candidate_precision - precisions["median"], 4),
            "category_accuracy": round(candidate_category - categories["median"], 4),
        },
        "code_fingerprints": sorted(fingerprints),
        "documents_losing_recall": regressed,
        "documents_missing_from_candidate": missing_documents,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_metrics(verdict: Dict[str, Any]) -> Dict[str, float]:
    """Flatten a verdict into the numeric series MLflow can plot across gates.

    Pure. Booleans become 0/1 so a failed check is visible as a metric, and the
    per-document lists become counts -- their contents stay in the verdict.
    """
    metrics = {
        f"candidate_{name}": value
        for name, value in verdict["candidate"].items()
        if isinstance(value, float)
    }
    metrics.update(
        {f"delta_{name}": value for name, value in verdict["deltas_vs_baseline_median"].items()}
    )
    for name, spread in verdict["baseline_spread"].items():
        metrics.update({f"baseline_{name}_{stat}": value for stat, value in spread.items()})
    metrics.update({f"check_{name}": float(ok) for name, ok in verdict["checks"].items()})
    metrics["passed"] = float(verdict["passed"])
    metrics["spread_measurable"] = float(verdict["spread_measurable"])
    metrics["recall_margin_over_best_baseline"] = verdict["recall_margin_over_best_baseline"]
    metrics["documents_losing_recall"] = float(len(verdict["documents_losing_recall"]))
    metrics["documents_missing_from_candidate"] = float(
        len(verdict["documents_missing_from_candidate"])
    )
    return metrics


def log_verdict(verdict: Dict[str, Any], config: Any, report_path: Optional[Path]) -> None:
    """Record the verdict as one MLflow run. No-ops when tracking is off, and
    never changes the exit status this script reports the verdict through."""
    with tracked_run(config, MLFLOW_STAGE), non_fatal("gate verdict log"):
        log_params({
            "tolerance": verdict["tolerance"],
            "min_rooms_lost": verdict["min_rooms_lost"],
            "baselines": verdict["baselines"],
            "noise_floor": verdict["noise_floor"],
            "candidate_run_dir": verdict["candidate"]["run_dir"],
        })
        log_metrics(MLFLOW_STAGE, gate_metrics(verdict))
        if report_path is not None:
            log_artifact(str(report_path))


GT_EVAL = "room_gt_eval"


def scored_report(report: Dict[str, Any], section: str) -> Dict[str, Any]:
    """The scored part of either report kind, and which kind it was.

    `eval/room_gt_eval.py` writes the scores at the top level. `eval/room_replay.py`
    nests several scorings of one run -- `as_run` (the filter settings the run
    itself used) and `baseline`/`variant` (replayed under current code) -- and
    only the replayed ones reflect code changes made since the run.

    Returns (scores, kind); `kind` lets the caller refuse a mixed comparison.
    """
    if "localization" in report:
        return report, GT_EVAL
    if section not in report:
        raise SystemExit(
            f"report has neither top-level scores nor a {section!r} section; "
            f"has {sorted(report)[:6]}"
        )
    # run_dir and the producing code's fingerprint live at the replay report's
    # top level; the verdict identifies the candidate by one and checks
    # comparability by the other, so carry both into the section.
    return {
        "run_dir": report.get("run_dir"),
        "run_code_fingerprint": report.get("run_code_fingerprint", "unknown"),
        **report[section],
    }, section


def load_reports(paths: Sequence[Path], section: str) -> List[Dict[str, Any]]:
    """Load reports of ONE kind; a mixed set is refused.

    Comparing a candidate replayed under current code against baselines scored
    before a pipeline change is how a fix to the pipeline gets read as a win
    for the model. Measured case: the 2026-10-28 merge fix lifted every
    baseline's recall by ~.025, so pre-fix baselines understate the bar.
    """
    loaded = [scored_report(json.loads(path.read_text()), section) for path in paths]
    kinds = {kind for _, kind in loaded}
    if len(kinds) > 1:
        raise SystemExit(
            f"refusing to compare reports of different kinds ({sorted(kinds)}): "
            "score every run the same way."
        )
    return [scores for scores, _ in loaded]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply the D8 gate to a fine-tuned run")
    parser.add_argument("--baseline", type=Path, action="append", required=True,
                        dest="baselines", help="room_gt_eval report for a baseline run; repeat")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="room_gt_eval report for the adapter run")
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="Allowed drop in precision and category accuracy (default 0.01)")
    parser.add_argument("--min-rooms-lost", type=int, default=2,
                        help="A document regresses only if it falls below every "
                             "baseline AND loses at least this many GT rooms "
                             "(default 2; 1-room swings occur between baseline "
                             "seeds, so 1 would flag noise). 1 = the old strict rule.")
    parser.add_argument("--noise-floor", type=float, default=0.0,
                        help="How far above the best baseline the candidate's recall "
                             "must land (default 0.0). One baseline has a spread of "
                             "zero, so the recall check then passes on any positive "
                             "delta; pass the seed spread measured elsewhere as the "
                             "floor to compare a candidate against a single run.")
    parser.add_argument("--replay-section", default="baseline",
                        help="Which section of an eval/room_replay.py report to score "
                             "(default: baseline, i.e. replayed under current code). "
                             "Ignored for eval/room_gt_eval.py reports, which score at "
                             "the top level; the two kinds cannot be mixed.")
    parser.add_argument("--out", type=Path, default=None, help="Also write the verdict here")
    add_mlflow_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    reports = load_reports(list(args.baselines) + [args.candidate], args.replay_section)
    verdict = apply_gate(reports[:-1], reports[-1], args.tolerance, args.min_rooms_lost,
                         args.noise_floor)
    verdict["replay_section"] = args.replay_section
    emit_report(verdict, args.out)
    log_verdict(verdict, mlflow_config_from_args(args), args.out)
    sys.exit(0 if verdict["passed"] else 1)
