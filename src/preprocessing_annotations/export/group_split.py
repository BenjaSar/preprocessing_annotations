"""Group-aware train/val/test assignment.

Splitting by page puts near-duplicate pages of one document on both sides of
the split (measured on sprint1_verify54: page thumbnails within a document
correlate .80-.93). Splitting by document is not enough either: the AVI-ON and
Electrical PDFs of one building share room names (Jaccard .50-.59), and two
schools of one design family reach the same level (.39-.45) against a
.029 median for unrelated documents.

So the split unit is a *group*: by default one source document, and optionally
a wider family named in a group map. Groups never straddle splits, and groups
pinned in the map always land in test (used to keep the room-GT pages out of
any training set, since they are the only room evaluation set there is).

Group sizes are uneven (1-8 pages per GT document), so slicing a shuffled list
would miss the ratios badly. Assignment is deficit-greedy on *item* counts:
groups are visited largest first (seeded shuffle breaks ties) and each goes to
whichever split is furthest below its target share.
"""

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

logger = logging.getLogger(__name__)

TRAIN, VAL, TEST = "train", "val", "test"
SPLIT_NAMES: Tuple[str, str, str] = (TRAIN, VAL, TEST)

_PAGE_SUFFIX = re.compile(r"_page\d+$")


def document_key(annotation_file_name: str) -> str:
    """Annotation/image file name -> source document key (page index stripped)."""
    return _PAGE_SUFFIX.sub("", Path(annotation_file_name).stem)


@dataclass(frozen=True)
class GroupMap:
    """Document -> group name, plus the groups that always go to test.

    A document with no entry is its own group, so an empty map means
    "split by document".
    """

    groups: Mapping[str, str] = field(default_factory=dict)
    pinned_test_groups: frozenset = frozenset()

    @classmethod
    def load(cls, path: Path | str | None) -> "GroupMap":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"split group map not found: {path}. Write one with "
                f"scripts/make_split_group_map.py, or set "
                f"ExportConfig.split_group_map_path to None to split by document."
            )
        data = json.loads(path.read_text())
        return cls(
            groups=dict(data.get("groups", {})),
            pinned_test_groups=frozenset(data.get("pinned_test_groups", ())),
        )

    def group_of(self, annotation_file_name: str) -> str:
        document = document_key(annotation_file_name)
        return self.groups.get(document, document)


def assign_groups(
    group_sizes: Mapping[str, int], ratios: Sequence[float], seed: int,
    pinned_test_groups: Iterable[str] = (),
) -> Dict[str, str]:
    """Group name -> split name, deficit-greedy on item counts."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {tuple(ratios)}")
    pinned = {g for g in pinned_test_groups if g in group_sizes}
    assignment = {group: TEST for group in pinned}
    counts = {TRAIN: 0, VAL: 0, TEST: sum(group_sizes[g] for g in pinned)}

    total = sum(group_sizes.values())
    targets = dict(zip(SPLIT_NAMES, (ratio * total for ratio in ratios)))

    remaining = [g for g in group_sizes if g not in pinned]
    random.Random(seed).shuffle(remaining)
    remaining.sort(key=lambda g: group_sizes[g], reverse=True)
    for group in remaining:
        split = max(SPLIT_NAMES, key=lambda s: targets[s] - counts[s])
        assignment[group] = split
        counts[split] += group_sizes[group]

    empty = [s for s in SPLIT_NAMES if counts[s] == 0]
    if empty:
        logger.warning(
            "Group split left %s empty: %d group(s) over %d item(s) cannot fill "
            "ratios %s without splitting a group",
            "/".join(empty), len(group_sizes), total, tuple(ratios),
        )
    return assignment


def split_items(
    item_groups: Mapping[str, str], ratios: Sequence[float], seed: int,
    pinned_test_groups: Iterable[str] = (),
) -> Dict[str, str]:
    """Item name -> split name, via each item's group."""
    sizes: Dict[str, int] = {}
    for group in item_groups.values():
        sizes[group] = sizes.get(group, 0) + 1
    by_group = assign_groups(sizes, ratios, seed, pinned_test_groups)
    return {item: by_group[group] for item, group in item_groups.items()}
