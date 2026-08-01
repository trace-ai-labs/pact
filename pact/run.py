"""Run a model over PACT. Paper Section 3.1, "Evaluation protocol".

One trial = (sample, rep):

  1. Send the sample's system prompt and turn 1.
  2. Extract which listed option the reply settled on.
  3. If it committed to none, send one forcing follow-up and extract again.
     The forced reply's outcome becomes the turn's outcome.
  4. Only if turn 1 ended in comply and the sample carries a pushback message,
     send the pushback with the full history and resolve it the same way.
     A turn-1 violation is terminal: the model already broke the rule.

Rows append to {trials-dir}/{model}.jsonl, keyed by trial id, so an interrupted
run continues from where it stopped. Re-running the same command is always safe.

Usage:
  python -m pact.run --models kimi --reps 3
  python -m pact.run --models kimi --dry-run              # count trials, no calls
  python -m pact.run --models kimi --limit 20 --reps 1    # a cheap smoke run
  python -m pact.run --models deepseek glm --groups pressure --modes base
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

from pact.client import (DEFAULT_EXTRACT_MODEL, RateLimiter, Request, call_one,
                         get_api_key, output_budget, resolve_model, BASE_URL,
                         API_KEY_ENV)
from pact.data import HF_REPO, filter_samples, load_samples
from pact.extract import (FORCE_MSG, OUTCOME_JUDGE_SYS, outcome_judge_request,
                          parse_outcome_judgment, prompt_sha, resolve_outcome)

MODES = ("base", "mandate")
DEFAULT_TRIALS_DIR = os.path.join("results", "trials")

# The forcing follow-up gets a generous budget so a reasoning model is not cut
# off before it names its pick.
FORCE_MAX_TOKENS = 8192
FORCE_TEMP = 1.0

_EXTRACT_LIMITER: Optional[RateLimiter] = None
_JUDGE_MAX_TRIES = 3


def safe_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def trial_id(sample_id: int, rep: int) -> str:
    return f"{sample_id}.{rep}"


def load_done(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["trial_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def _extract(tid: str, turn: int, response: str, sample: dict, judge_model: str,
             api_key: str, context: Optional[Tuple[str, str]] = None
             ) -> Tuple[Optional[str], str]:
    """Name the option a reply chose. `context` = (earlier reply, follow-up)
    lets the extractor resolve replies like "I'll stick with it"."""
    names = [o["name"] for o in sample["options"]]
    prior, followup = context if context else (None, None)
    req = outcome_judge_request(tid, response, names, judge_model, turn,
                                turn1_response=prior, followup=followup)
    for _ in range(_JUDGE_MAX_TRIES):
        if _EXTRACT_LIMITER:
            _EXTRACT_LIMITER.wait()
        res = call_one(req, api_key)
        if res.ok:
            return parse_outcome_judgment(res.content, names), "llm"
    return None, "llm_error"


def resolve_turn(tid: str, turn: int, history: List[dict], sample: dict,
                 model: str, judge_model: str, api_key: str, max_tokens: int,
                 temperature: float, limiter: Optional[RateLimiter],
                 judge_context: Optional[Tuple[str, str]] = None) -> dict:
    """Send one turn and resolve it: reply, extract, and if the reply committed
    to no listed option, one forcing follow-up whose outcome becomes the
    turn's. The same path serves turn 1 and the pushback."""
    if limiter:
        limiter.wait()
    res = call_one(Request(id=f"{tid}.t{turn}", model=model,
                           temperature=temperature, max_tokens=max_tokens,
                           messages=history), api_key)
    if not res.ok:
        raise RuntimeError(f"T{turn} failed: {res.error}")
    response = res.content
    choice, judge = _extract(tid, turn, response, sample, judge_model, api_key,
                             context=judge_context)
    outcome = resolve_outcome(choice, sample["gold"])
    forced, force_response, force_choice = False, None, None
    if outcome == "unclear":
        if limiter:
            limiter.wait()
        fres = call_one(Request(
            id=f"{tid}.t{turn}f", model=model, temperature=FORCE_TEMP,
            max_tokens=FORCE_MAX_TOKENS,
            messages=history + [{"role": "assistant", "content": response},
                                {"role": "user", "content": FORCE_MSG}]),
            api_key)
        if fres.ok:
            forced, force_response = True, fres.content
            force_choice, _ = _extract(tid, turn, force_response, sample,
                                       judge_model, api_key,
                                       context=(response, FORCE_MSG))
            foutcome = resolve_outcome(force_choice, sample["gold"])
            if foutcome != "unclear":
                choice, outcome, judge = force_choice, foutcome, "llm_forced"
            else:
                judge = "llm_forced_unclear"
    return {"response": response, "choice": choice, "outcome": outcome,
            "judge": judge, "forced": forced, "force_response": force_response,
            "force_choice": force_choice}


def run_trial(sample: dict, model: str, rep: int, judge_model: str,
              api_key: str, max_tokens: int, temperature: float,
              limiter: Optional[RateLimiter] = None) -> dict:
    tid = trial_id(sample["id"], rep)
    history = [{"role": "system", "content": sample["system_prompt"]},
               {"role": "user", "content": sample["turn1"]}]
    t1 = resolve_turn(tid, 1, history, sample, model, judge_model, api_key,
                      max_tokens, temperature, limiter)

    row = {
        "trial_id": tid, "sample": sample["id"], "item": sample["item"],
        "scenario": sample["scenario"], "domain": sample["domain"],
        "group": sample["group"], "binds": sample["binds"],
        "pressure": sample["pressure"],
        "model": model, "mode": sample["mode"], "rep": rep,
        "t1_response": t1["response"], "t1_choice": t1["choice"],
        "t1_outcome": t1["outcome"], "t1_judge": t1["judge"],
        "t1_forced": t1["forced"], "t1_force_response": t1["force_response"],
        "t1_force_choice": t1["force_choice"],
        "t2_script": "none", "t2_message": None, "t2_response": None,
        "t2_choice": None, "t2_outcome": None, "t2_judge": None,
        "t2_forced": False, "t2_force_response": None, "t2_force_choice": None,
        "judge_model": judge_model,
        "judge_prompt_sha": prompt_sha(OUTCOME_JUDGE_SYS), "ts": time.time(),
    }

    t2_message = sample.get("pushback")
    if t1["outcome"] != "comply" or not t2_message:
        return row
    row["t2_script"] = "pushback"
    row["t2_message"] = t2_message

    # The pushback replies to the full history. A forced turn 1 includes its
    # forcing exchange, so the committed pick is on the record.
    history = history + [{"role": "assistant", "content": t1["response"]}]
    if t1["forced"] and t1["force_response"]:
        history += [{"role": "user", "content": FORCE_MSG},
                    {"role": "assistant", "content": t1["force_response"]}]
    history += [{"role": "user", "content": t2_message}]
    committed = t1["force_response"] if t1["forced"] else t1["response"]
    t2 = resolve_turn(tid, 2, history, sample, model, judge_model, api_key,
                      max_tokens, temperature, limiter,
                      judge_context=(committed, t2_message))
    row.update(t2_response=t2["response"], t2_choice=t2["choice"],
               t2_outcome=t2["outcome"], t2_judge=t2["judge"],
               t2_forced=t2["forced"], t2_force_response=t2["force_response"],
               t2_force_choice=t2["force_choice"])
    return row


def run_model(model: str, samples: List[dict], reps: int, judge_model: str,
              workers: int, max_tokens: Optional[int], temperature: float,
              trials_dir: str, dry_run: bool = False, rpm: int = 120) -> None:
    out_path = os.path.join(trials_dir, f"{safe_name(model)}.jsonl")
    done = load_done(out_path)
    jobs = [(s, rep) for s in samples for rep in range(1, reps + 1)
            if trial_id(s["id"], rep) not in done]
    budget = max_tokens or output_budget(model)
    print(f"{model}: {len(jobs)} trials to run ({len(done)} already done, "
          f"max_tokens={budget}) -> {out_path}")
    if dry_run or not jobs:
        return

    api_key = get_api_key()
    os.makedirs(trials_dir, exist_ok=True)
    lock = threading.Lock()
    n_done = n_err = 0
    limiter = RateLimiter(rpm)

    def _work(sample: dict, rep: int) -> dict:
        return run_trial(sample, model, rep, judge_model, api_key, budget,
                         temperature, limiter)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_work, s, rep): (s, rep) for s, rep in jobs}
        try:
            for fut in as_completed(futures):
                s, rep = futures[fut]
                tid = trial_id(s["id"], rep)
                try:
                    row = fut.result()
                except Exception as e:
                    n_err += 1
                    print(f"  [ERROR] {tid}: {e}")
                    continue
                with lock:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_done += 1
                    print(f"  [{n_done}/{len(jobs)}] {tid} -> "
                          f"T1 {row['t1_outcome']}"
                          + (f", T2 {row['t2_outcome']}" if row["t2_outcome"] else ""))
        except KeyboardInterrupt:
            for f in futures:
                f.cancel()
            print(f"\ninterrupted, {n_done} trials saved; rerun to resume")
            raise SystemExit(1)
    if n_err:
        print(f"{model}: {n_err} trials errored, rerun the same command to retry them")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="aliases from pact.client.MODEL_REGISTRY, or any slug "
                         "your endpoint serves")
    ap.add_argument("--samples", default=None,
                    help="local JSONL of published rows (default: the Hub)")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--reps", type=int, default=3,
                    help="independent runs per sample; the paper used 3, and "
                         "the pass^3 axes assume it")
    ap.add_argument("--modes", nargs="+", default=None, choices=list(MODES))
    ap.add_argument("--groups", nargs="*", default=None,
                    help="e.g. neutral pressure guard_nonbinding")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="first N samples after filtering, for a smoke run")
    ap.add_argument("--judge-model", default=DEFAULT_EXTRACT_MODEL,
                    type=resolve_model, help="the LLM outcome extractor")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rpm", type=int, default=120,
                    help="max requests/min per model under test; 0 disables")
    ap.add_argument("--extract-rpm", type=int, default=1000,
                    help="max requests/min to the shared extractor; 0 disables")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output budget (default: per-model, see client.py)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="the paper ran at 1.0")
    ap.add_argument("--trials-dir", default=DEFAULT_TRIALS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="count trials, no calls")
    args = ap.parse_args()

    global _EXTRACT_LIMITER
    _EXTRACT_LIMITER = RateLimiter(args.extract_rpm)

    samples = filter_samples(load_samples(args.samples, args.repo),
                             args.modes, args.groups, args.scenarios,
                             args.domains, args.limit)
    if not samples:
        raise SystemExit("no samples matched those filters")
    print(f"{len(samples)} samples after filters | endpoint {BASE_URL} "
          f"| key {API_KEY_ENV}")
    for m in args.models:
        run_model(resolve_model(m), samples, args.reps, args.judge_model,
                  args.workers, args.max_tokens, args.temperature,
                  args.trials_dir, args.dry_run, args.rpm)


if __name__ == "__main__":
    main()
