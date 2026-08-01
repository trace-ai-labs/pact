"""Load the PACT dataset from the Hugging Face Hub.

The dataset is gated: accept the license at
https://huggingface.co/datasets/trace-ai-labs/pact and authenticate
(`huggingface-cli login`, or set HF_TOKEN) before the first load.

The published columns are flat and minimal. `load_samples` normalizes them into
the shape the runner and the metrics expect, deriving the three fields the
export leaves implicit:

  item    the scenario cell a base/mandate pair shares. Rows ship in pairs, the
          base sample at 2k and its mandate twin at 2k+1, so item = id // 2.
          Every paired metric (Steerability, and the mode split in PACTScore)
          keys off this.
  domain  the part of `scenario` before the dot.
  binds   whether the rule applies to this request: true on every group except
          the two `*_nonbinding` ones, where standing down is the correct call.

The dataset is not redistributed from here and this module writes nothing to
disk. `datasets` keeps its own local cache, so offline reruns work with
HF_HUB_OFFLINE=1 once you have loaded it a first time.

Usage:
  python -m pact.data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import List, Optional

HF_REPO = "trace-ai-labs/pact"
HF_SPLIT = "test"

# Groups where the embedded rule does NOT apply, so the correct answer is to
# stand down from it rather than enforce it.
NONBINDING_GROUPS = ("guard_nonbinding", "attack_nonbinding")

# Only these groups carry a scripted pushback turn.
PUSHBACK_GROUPS = ("neutral", "pressure")

PUBLISHED_COLUMNS = ("id", "system_prompt", "turn1", "turn2", "option_names",
                     "gold_compliant_choices", "mode", "group", "pressure",
                     "scenario")


def normalize(row: dict) -> dict:
    """One published row -> the sample dict used throughout this package."""
    gold = list(row["gold_compliant_choices"])
    names = list(row["option_names"])
    rid = int(row["id"])
    scenario = row["scenario"]
    return {
        "id": rid,
        "item": rid // 2,
        "scenario": scenario,
        "domain": scenario.split(".")[0],
        "group": row["group"],
        "binds": row["group"] not in NONBINDING_GROUPS,
        "pressure": row.get("pressure", "none"),
        "mode": row["mode"],
        "system_prompt": row["system_prompt"],
        "turn1": row["turn1"],
        "pushback": row.get("turn2"),
        "options": [{"name": n, "compliant": n in gold} for n in names],
        "gold": {"compliant_choices": gold},
    }


def load_published(path: Optional[str] = None, repo: str = HF_REPO,
                   split: str = HF_SPLIT) -> List[dict]:
    """The published rows, exactly as they ship. Reads a local JSONL when
    `path` is given, otherwise pulls from the Hub."""
    if path:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "pip install datasets  (or pass --samples with a local JSONL)")
    try:
        ds = load_dataset(repo, split=split)
    except Exception as e:
        raise SystemExit(
            f"could not load {repo}:{split} - {type(e).__name__}: {e}\n\n"
            f"The dataset is gated. Accept the license at\n"
            f"  https://huggingface.co/datasets/{repo}\n"
            f"then authenticate with `huggingface-cli login` or set HF_TOKEN.")
    return [{k: r[k] for k in PUBLISHED_COLUMNS} for r in ds]


def load_samples(path: Optional[str] = None, repo: str = HF_REPO,
                 split: str = HF_SPLIT) -> List[dict]:
    """All 3,364 samples, normalized and sorted by published row id."""
    rows = load_published(path, repo, split)
    return sorted((normalize(r) for r in rows), key=lambda s: s["id"])


def filter_samples(samples: List[dict], modes: Optional[List[str]] = None,
                   groups: Optional[List[str]] = None,
                   scenarios: Optional[List[str]] = None,
                   domains: Optional[List[str]] = None,
                   limit: int = 0) -> List[dict]:
    out = samples
    if modes:
        out = [s for s in out if s["mode"] in set(modes)]
    if groups:
        out = [s for s in out if s["group"] in set(groups)]
    if scenarios:
        out = [s for s in out if s["scenario"] in set(scenarios)]
    if domains:
        out = [s for s in out if s["domain"] in set(domains)]
    if limit:
        out = out[:limit]
    return out


def _stats(samples: List[dict]) -> None:
    cells = {s["item"] for s in samples}
    print(f"{len(samples)} samples | {len(cells)} cells "
          f"| {len({s['scenario'] for s in samples})} scenarios "
          f"| {len({s['domain'] for s in samples})} domains")
    for f in ("mode", "group"):
        print(f"  {f}: {dict(Counter(s[f] for s in samples))}")
    print(f"  pressure: {dict(Counter(s['pressure'] for s in samples))}")
    print(f"  carrying a pushback turn: "
          f"{sum(1 for s in samples if s['pushback'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default=None,
                    help="a local JSONL of published rows you already have, "
                         "instead of the Hub")
    ap.add_argument("--repo", default=HF_REPO)
    args = ap.parse_args()

    rows = load_published(args.samples, args.repo)
    _stats(sorted((normalize(r) for r in rows), key=lambda s: s["id"]))


if __name__ == "__main__":
    main()
