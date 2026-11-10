"""No document the room GT scores may appear in SFT training data.

Every claim the project makes about a fine-tune -- .6112, .6132, every gate
verdict -- rests on the adapter never having seen the pages it is scored on.
Nothing enforced that. The SFT datasets are exported from the corpus batches
and the GT is scored on test_input/, and those two sets were kept apart by
where the files happened to live, not by a check. One export pointed at the
wrong run directory and every gate in the project silently becomes a
memorisation score.

Measured 2026-11-09 over the five cap-200 datasets the sft_cap200_v1 training
used: 120 distinct documents across train/val/test, 13 GT-scored documents,
intersection empty. This module turns that measurement into a check that can
be re-run before a training job instead of trusted.

Document identity is `export.group_split.document_key` -- the same key the
COCO split uses -- so a page and its document agree about what they belong to
here and there. Comparison is case-insensitive because the GT corpus names
PDFs in title case ("BP-1 Violet Elementary School Electrical Pages.pdf") and
the pipeline writes page images in lower snake case.

Usage:
    python eval/sft_holdout.py --dataset <dir> [--dataset <dir>...] \\
        --evaluation-input <dir with the GT-scored PDFs>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "eval"))

from preprocessing_annotations.export.group_split import document_key

from report_io import emit_report

logger = logging.getLogger(__name__)

SPLIT_FILES = ("train.jsonl", "val.jsonl", "test.jsonl")


def _key(name: str) -> str:
    """Document identity, case-folded. `document_key` strips the page suffix;
    folding case lets a title-case PDF match its lower-snake-case pages."""
    return document_key(name).casefold()


def evaluation_documents(evaluation_input: Path) -> Set[str]:
    """The documents the room GT scores: one per PDF in the eval input."""
    return {_key(pdf.name) for pdf in sorted(evaluation_input.glob("*.pdf"))}


def dataset_documents(dataset_dir: Path) -> Dict[str, Set[str]]:
    """Document key -> the split files it appears in, for one SFT dataset.

    Reads `metadata.source_page`, the full path of the page a tile was cut
    from; `metadata.image` is the tile crop, whose name carries the tile's
    pixel offsets and so does not reduce to a document on its own.
    """
    found: Dict[str, Set[str]] = {}
    for split in SPLIT_FILES:
        path = dataset_dir / split
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line).get("metadata") or {}
            source = metadata.get("source_page") or metadata.get("image")
            if source:
                found.setdefault(_key(Path(source).name), set()).add(split)
    return found


def check(datasets: Sequence[Path], evaluation_input: Path) -> dict:
    """Pure-ish (reads files, returns a verdict): is the GT set held out?

    A dataset with no split files is reported as empty rather than passed over
    silently -- an export that wrote nothing must not read as a clean holdout.
    """
    scored = evaluation_documents(evaluation_input)
    if not scored:
        raise ValueError(f"no PDFs to score in {evaluation_input}")

    per_dataset: Dict[str, Dict[str, Set[str]]] = {
        str(dataset): dataset_documents(dataset) for dataset in datasets
    }
    leaks: List[dict] = []
    for dataset, documents in per_dataset.items():
        for document in sorted(set(documents) & scored):
            leaks.append({
                "dataset": dataset,
                "document": document,
                "splits": sorted(documents[document]),
            })

    all_documents: Set[str] = set()
    for documents in per_dataset.values():
        all_documents |= set(documents)

    return {
        "evaluation_input": str(evaluation_input),
        "evaluation_documents": len(scored),
        "training_documents": len(all_documents),
        "empty_datasets": sorted(d for d, docs in per_dataset.items() if not docs),
        "leaked_documents": leaks,
        "held_out": not leaks,
    }


def _format(verdict: dict) -> Iterable[str]:
    yield (f"{verdict['training_documents']} training documents vs "
           f"{verdict['evaluation_documents']} GT-scored documents")
    for empty in verdict["empty_datasets"]:
        yield f"  WARNING empty dataset (no split files read): {empty}"
    for leak in verdict["leaked_documents"]:
        yield (f"  LEAK {leak['document']} in {leak['splits']} of {leak['dataset']}")
    yield "held out" if verdict["held_out"] else "NOT held out"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        dest="datasets", help="SFT dataset directory; repeat")
    parser.add_argument("--evaluation-input", type=Path, required=True,
                        help="Directory of the PDFs the room GT scores")
    parser.add_argument("--out", type=Path, default=None, help="Also write the verdict here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    verdict = check(args.datasets, args.evaluation_input)
    for line in _format(verdict):
        logger.info(line)
    emit_report(verdict, args.out)
    sys.exit(0 if verdict["held_out"] else 1)
