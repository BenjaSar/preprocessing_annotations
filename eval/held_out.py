"""Leave-one-document-out selection of a tunable value.

A filter threshold tuned and scored on the same GT is in-sample evidence. This
selects per held-out document: choose, on the other documents, the candidate
that locates the most GT rooms among those whose precision is not below the
default's, then score that choice on the held-out document. The summed
held-out totals are the out-of-sample estimate of "pick the best candidate".

Tie-break: earlier candidate wins, so list the default first.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from room_gt_eval import LocalizationScore


@dataclass(frozen=True)
class Fold:
    held_out: str
    chosen: Any
    located: int
    predictions: int
    predictions_in_gt: int


@dataclass(frozen=True)
class HeldOutResult:
    folds: List[Fold]
    located: int
    predictions: int
    predictions_in_gt: int
    default_located: int
    default_predictions: int
    default_predictions_in_gt: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "folds": [fold.__dict__ for fold in self.folds],
            "held_out": _rates(self.located, self.predictions, self.predictions_in_gt),
            "default": _rates(
                self.default_located, self.default_predictions,
                self.default_predictions_in_gt,
            ),
        }


def _rates(located: int, predictions: int, in_gt: int) -> Dict[str, Any]:
    return {
        "gt_rooms_located": located,
        "predictions": predictions,
        "label_precision": in_gt / predictions if predictions else 0.0,
    }


def _totals(
    scores: Mapping[str, LocalizationScore], documents: Sequence[str],
) -> Tuple[int, int, int]:
    """(gt_rooms_located, predictions, predictions_center_in_gt) over documents."""
    return (
        sum(scores[d].gt_rooms_located for d in documents),
        sum(scores[d].predictions for d in documents),
        sum(scores[d].predictions_center_in_gt for d in documents),
    )


def _precision_not_below(candidate: Tuple[int, int, int], reference: Tuple[int, int, int]) -> bool:
    _, c_pred, c_tp = candidate
    _, r_pred, r_tp = reference
    return c_tp * r_pred >= r_tp * c_pred


def leave_one_document_out(
    scores: Mapping[Any, Mapping[str, LocalizationScore]], default: Any,
) -> HeldOutResult:
    """scores[candidate][document]; candidates iterate in tie-break order."""
    documents = sorted(scores[default])
    folds: List[Fold] = []
    for held_out in documents:
        train = [d for d in documents if d != held_out]
        reference = _totals(scores[default], train)
        acceptable = [
            candidate for candidate in scores
            if _precision_not_below(_totals(scores[candidate], train), reference)
        ]
        chosen = max(acceptable, key=lambda c: _totals(scores[c], train)[0])
        located, predictions, in_gt = _totals(scores[chosen], [held_out])
        folds.append(Fold(held_out, chosen, located, predictions, in_gt))
    default_located, default_predictions, default_in_gt = _totals(scores[default], documents)
    return HeldOutResult(
        folds=folds,
        located=sum(f.located for f in folds),
        predictions=sum(f.predictions for f in folds),
        predictions_in_gt=sum(f.predictions_in_gt for f in folds),
        default_located=default_located,
        default_predictions=default_predictions,
        default_predictions_in_gt=default_in_gt,
    )
