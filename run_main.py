"""
run_main.py - main data collection of the LLM Loftus & Palmer replication.

Fully crossed design: videos x noise levels x all questions in config/questions.csv x
repetitions. Every call is independent (one user message, no history, store=False where the
API has it) and the calls run in a fixed-seed random order. Messages are built exactly as in
the pilot (llm_common.py; checked against run_pilot.py at start-up).

Models of several providers can be used (OpenAI, xAI, Anthropic, Google, DeepSeek): set MODEL
to an entry of MODELS and put that provider's API key in .env (see llm_providers.py). The
message content is identical for all providers; only the request format differs.

Each session starts with one unrecorded test call. The model version it returns names the
output files:
    results/main_{model}_{version}_{reasoning}.csv            one row per call
    results/main_{model}_{version}_{reasoning}_runinfo.json   parameters, sessions, input hashes
    results/main_{model}_{version}_{reasoning}_call_order.csv planned call order
The run stops if the version differs from an earlier session's (existing file for the same
model and reasoning setting) or changes during the run; versions are never mixed in a file.

Usage (from the project folder):
    .venv\\Scripts\\python run_main.py            # checks, test call, then asks you to type "yes"
    .venv\\Scripts\\python run_main.py --yes      # no prompt; only accepted once a test call for
                                                 # the same model version has been made before
    .venv\\Scripts\\python run_main.py --check    # input checks only, no API calls
    .venv\\Scripts\\python run_main.py --summary  # summary of the existing results, no API calls

Re-running is safe: every (stimulus, question, repetition, model, version, reasoning setting)
combination already in the results file is skipped, including invalid answers, so paid calls
are never repeated. The API key is read from .env and is never printed or logged.
"""

import argparse
import csv
import json
import os
import platform
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import llm_common as lc
import llm_providers as lp

# =============================================================================
# PARAMETERS
# =============================================================================
# Model (must be listed in MODELS below)
MODEL = "deepseek-flash"                 # a dated snapshot ID may be given here instead
REASONING_EFFORT = "low"              # translated per provider, see llm_providers.py
TEMPERATURE = None                    # None = not sent (model default), logged as "default"
IMAGE_DETAIL = "high"                 # must match the pilot
MAX_OUTPUT_TOKENS = 4000              # includes hidden reasoning tokens (not on xAI, see notes)

# Provider and prices (USD per 1M tokens; cost is computed at the full input price, so cached
# input makes the real bill lower). Verified 2026-10-11 on each provider's model/pricing page.
MODELS = {
    "gpt-6-luna": {"provider": "openai", "price_input": 0.10, "price_output": 0.50},
    "gpt-6-sol": {"provider": "openai", "price_input": 2.00, "price_output": 10.00},
    "gpt-6.1-sol": {"provider": "openai", "price_input": 2.00, "price_output": 10.00},
    # xAI: prompts < 200k tokens; reasoning cannot be turned off (effort low/medium/high/xhigh)
    "grok-4.7": {"provider": "xai", "price_input": 2.00, "price_output": 6.00},
    # Anthropic: prompts <= 100k tokens; adaptive thinking is always on, effort sets its depth
    "claude-haiku-5-5": {"provider": "anthropic", "price_input": 0.10, "price_output": 0.50},
    # Google: output price includes thinking tokens; thinking levels minimal/low/medium/high
    "gemini-3.5-flash-lite": {"provider": "google", "price_input": 0.30, "price_output": 2.50},
    # DeepSeek V4.1 Flash: off-peak price; peak hours (UTC, Mon-Fri) cost twice as much
    "deepseek-flash": {"provider": "deepseek", "price_input": 0.15, "price_output": 0.60,
                       "peak": {"price_input": 0.30, "price_output": 1.20,
                                "utc_hours": [[1, 4], [6, 10]], "weekdays_only": True}},
}
if MODEL not in MODELS:
    sys.exit(f"MODEL {MODEL!r} is not in MODELS; add its provider and prices first.")
PROVIDER = MODELS[MODEL]["provider"]
PRICE_INPUT_PER_1M = MODELS[MODEL]["price_input"]
PRICE_OUTPUT_PER_1M = MODELS[MODEL]["price_output"]

# Design
VIDEOS = ["V1", "V2", "V3", "V4", "V5"]
NOISE_SDS = [0, 25, 50, 80]
N_REPS = 5
RANDOM_SEED = 20261011                # fixes the call order
CONFIRMATORY_CODES = ["s1", "s2", "s3", "s4", "s5"]   # verb_rank = number in the code

# Safety limits (per session; the test call counts towards both)
MAX_CALLS = 900
MAX_COST_USD = 10.0
MAX_RETRIES = 3                       # retries per call after the first attempt
RETRY_WAITS_SEC = [5, 15, 45]         # wait before retry 1, 2, 3
MAX_CONSECUTIVE_FAILURES = 5          # calls that failed after all retries
REQUEST_TIMEOUT_SEC = 180
MIN_INPUT_TOKENS_PER_FRAME = 50       # below this the images were evidently not processed

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "stimuli" / "manifest.csv"
QUESTIONS_PATH = PROJECT_DIR / "config" / "questions.csv"
VERBS_PATH = PROJECT_DIR / "config" / "verbs.csv"
RAW_VIDEO_DIR = PROJECT_DIR / "videos_raw"
RESULTS_DIR = PROJECT_DIR / "results"
FILE_PREFIX = "main"

RESULT_FIELDS = [
    "llm_answer_raw", "llm_answer", "llm_invalid_response", "llm_answer_flag",
    "video_filename", "video_id", "noise_sd", "stimulus_id", "question_code", "question_text",
    "verb_phrase", "embedding_severity_z", "embedding_speed_z", "confirmatory_set", "verb_rank",
    "llm_model", "llm_model_version", "temperature", "reasoning_setting",
    "tokens_input", "tokens_input_cached", "tokens_output", "tokens_reasoning", "total_cost",
    "frames_sent", "fps", "image_detail", "input_raw",
    "datetime", "repetition_index", "call_order", "finish_status", "response_id", "latency_sec",
]
KEY_FIELDS = ["stimulus_id", "question_code", "repetition_index", "llm_model",
              "llm_model_version", "reasoning_setting"]

# Settings that must stay the same across sessions writing to the same results file
DESIGN = {
    "MODEL": MODEL, "PROVIDER": PROVIDER, "REASONING_EFFORT": REASONING_EFFORT,
    "TEMPERATURE": TEMPERATURE, "IMAGE_DETAIL": IMAGE_DETAIL,
    "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS, "VIDEOS": VIDEOS, "NOISE_SDS": NOISE_SDS,
    "N_REPS": N_REPS, "RANDOM_SEED": RANDOM_SEED, "CONFIRMATORY_CODES": CONFIRMATORY_CODES,
}
# Values assumed for settings that runs made before they existed did not record
DESIGN_DEFAULTS_FOR_OLDER_RUNS = {"PROVIDER": "openai"}
LIMITS = {
    "PRICES_PER_1M": MODELS[MODEL],
    "MAX_CALLS": MAX_CALLS, "MAX_COST_USD": MAX_COST_USD, "MAX_RETRIES": MAX_RETRIES,
    "RETRY_WAITS_SEC": RETRY_WAITS_SEC, "MAX_CONSECUTIVE_FAILURES": MAX_CONSECUTIVE_FAILURES,
    "REQUEST_TIMEOUT_SEC": REQUEST_TIMEOUT_SEC,
    "MIN_INPUT_TOKENS_PER_FRAME": MIN_INPUT_TOKENS_PER_FRAME,
}
INPUT_FILES = {"questions": QUESTIONS_PATH, "verbs": VERBS_PATH, "manifest": MANIFEST_PATH}


# =============================================================================
# Input validation and design
# =============================================================================
def rel(path):
    """Path relative to the project folder for printing (full path if outside it)."""
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


def fail(problems, title):
    sys.exit(f"{title}:\n  " + "\n  ".join(problems))


def validate_inputs():
    """Questions, verb metadata and stimuli. Returns (questions, verb metadata by code, manifest)."""
    problems = []
    questions = lc.read_csv(QUESTIONS_PATH)
    codes = [q.get("question_code", "").strip() for q in questions]
    if not questions or {"question_code", "question_text"} - set(questions[0]):
        fail(["needs columns question_code, question_text"], QUESTIONS_PATH.name)
    for c in sorted({c for c in codes if codes.count(c) > 1}):
        problems.append(f"{QUESTIONS_PATH.name}: question code {c!r} appears more than once")
    for q in questions:
        if not q["question_code"].strip() or not q["question_text"].strip():
            problems.append(f"{QUESTIONS_PATH.name}: empty code or text in row {q}")
    for c in CONFIRMATORY_CODES:
        if c not in codes:
            problems.append(f"confirmatory code {c} is not in {QUESTIONS_PATH.name}")
        if not re.fullmatch(r"s\d+", c):
            problems.append(f"confirmatory code {c} has no rank number (expected s1, s2, ...)")

    verbs = lc.read_csv(VERBS_PATH)
    need = {"code", "verb_phrase", "embedding_severity_z", "embedding_speed_z"}
    if not verbs or need - set(verbs[0]):
        fail([f"needs columns {sorted(need)}"], VERBS_PATH.name)
    meta = {}
    for q in questions:
        code = q["question_code"]
        matches = [v for v in verbs if v["code"].strip() == code]
        if len(matches) != 1:
            problems.append(f"question code {code}: {len(matches)} matching rows in "
                            f"{VERBS_PATH.name} (needs exactly 1)")
            continue
        v = matches[0]
        for col in ("embedding_severity_z", "embedding_speed_z"):
            val = v[col].strip()
            if val:
                try:
                    float(val)
                except ValueError:
                    problems.append(f"{code}: {col} = {val!r} is not a number")
            elif code in CONFIRMATORY_CODES:
                problems.append(f"{code}: {col} is missing (required for {CONFIRMATORY_CODES})")
        if v["verb_phrase"].strip().lower() not in q["question_text"].lower():
            problems.append(f"{code}: verb phrase {v['verb_phrase']!r} does not occur in its "
                            f"question text - codes in the two files may be mismatched")
        conf = code in CONFIRMATORY_CODES
        meta[code] = {
            "verb_phrase": v["verb_phrase"].strip(),
            "embedding_severity_z": v["embedding_severity_z"].strip(),
            "embedding_speed_z": v["embedding_speed_z"].strip(),
            "confirmatory_set": "TRUE" if conf else "FALSE",
            "verb_rank": int(code[1:]) if conf else "",
        }

    manifest = lc.read_csv(MANIFEST_PATH)
    for row in manifest:                      # every stimulus, not only the ones in the design
        lc.load_frames(PROJECT_DIR, row)      # exits if frames are missing
    have = {(r["video_id"], int(r["noise_sd"])) for r in manifest}
    for v in VIDEOS:
        for sd in NOISE_SDS:
            if (v, sd) not in have:
                problems.append(f"no stimulus for {v} with noise SD {sd} in {MANIFEST_PATH.name}")
    sources = lc.source_video_filenames(RAW_VIDEO_DIR)
    for v in VIDEOS:
        if v not in sources:
            problems.append(f"no source video for {v} in {RAW_VIDEO_DIR.name}/")
    if problems:
        fail(problems, "Input check failed - no API calls were made")
    return questions, meta, manifest


def build_plan(questions, meta, manifest):
    by_key = {(r["video_id"], int(r["noise_sd"])): r for r in manifest}
    sources = lc.source_video_filenames(RAW_VIDEO_DIR)
    plan = []
    for video in VIDEOS:
        for sd in NOISE_SDS:
            stim = by_key[(video, sd)]
            for q in questions:
                for rep in range(1, N_REPS + 1):
                    plan.append({"stim": stim, "video_filename": sources[video],
                                 "question_code": q["question_code"],
                                 "question_text": q["question_text"], "repetition_index": rep,
                                 "meta": meta[q["question_code"]]})
    random.Random(RANDOM_SEED).shuffle(plan)
    for i, p in enumerate(plan, start=1):
        p["call_order"] = i
    return plan


def check_identical_to_pilot(plan):
    """Every (stimulus, question) message must equal the one run_pilot.py would build."""
    import run_pilot
    for name in ("PROMPT_INTRO", "FRAME_LABEL", "PROMPT_QUESTION"):
        if getattr(run_pilot, name) != getattr(lc, name):
            sys.exit(f"{name} differs from run_pilot.py - messages would not match the pilot.")
    if run_pilot.IMAGE_DETAIL != IMAGE_DETAIL:
        sys.exit(f"IMAGE_DETAIL {IMAGE_DETAIL!r} differs from the pilot's {run_pilot.IMAGE_DETAIL!r}.")

    seen = {}
    for p in plan:
        seen.setdefault((p["stim"]["stimulus_id"], p["question_code"]), p)
    for (sid, code), p in sorted(seen.items()):
        if run_pilot.build_message(p) != lc.build_message(p, PROJECT_DIR, IMAGE_DETAIL):
            sys.exit(f"Message for {sid} / {code} differs from the pilot's. Stopping.")
        run_pilot._b64_cache.clear()          # keep memory use low
    print(f"Message check: all {len(seen)} stimulus x question messages are identical to the pilot's.")

    # The provider's request must carry exactly these texts and frames, in this order.
    # (No API call: the client is created with a placeholder key and never used here.)
    adapter = lp.make_adapter(PROVIDER, "placeholder-key-for-offline-check", REQUEST_TIMEOUT_SEC)
    one_per_stimulus = {}
    for (sid, code), p in sorted(seen.items()):
        one_per_stimulus.setdefault(sid, p)
    for sid, p in one_per_stimulus.items():
        parts, *_ = lc.build_parts(p, PROJECT_DIR)
        req = adapter.build_request(MODEL, parts, reasoning_effort=REASONING_EFFORT,
                                    max_output_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE,
                                    image_detail=IMAGE_DETAIL)
        expected = [("text", x["text"]) if x["type"] == "text" else ("image", x["path"].read_bytes())
                    for x in parts]
        if adapter.content_signature(req) != expected:
            sys.exit(f"The {PROVIDER} request for {sid} does not carry the pilot's message. Stopping.")
        if PROVIDER in ("openai", "xai") and \
                req["input"][0]["content"] != run_pilot.build_message(p)[0]:
            sys.exit(f"The {PROVIDER} request for {sid} differs from the pilot's payload. Stopping.")
        run_pilot._b64_cache.clear()
    print(f"Provider check ({PROVIDER}): the request carries the same {len(expected)} message parts "
          f"(texts and frames, same order) for all {len(one_per_stimulus)} stimuli.")


# =============================================================================
# Output files
# =============================================================================
def output_paths(version):
    stem = "_".join(lc.safe_name(x) for x in (FILE_PREFIX, MODEL, version, REASONING_EFFORT))
    return {
        "results": RESULTS_DIR / f"{stem}.csv",
        "runinfo": RESULTS_DIR / f"{stem}_runinfo.json",
        "call_order": RESULTS_DIR / f"{stem}_call_order.csv",
        "mismatch": RESULTS_DIR / f"{stem}_version_mismatch.csv",
    }


def existing_versions():
    """{version: results path} of main results files for this MODEL and REASONING_EFFORT."""
    out = {}
    for p in RESULTS_DIR.glob(f"{FILE_PREFIX}_*_runinfo.json"):
        info = json.loads(p.read_text(encoding="utf-8"))
        if info["design"]["MODEL"] == MODEL and info["design"]["REASONING_EFFORT"] == REASONING_EFFORT:
            out[info["model_version"]] = p
    return out


def check_previous_sessions(hashes):
    """Before any paid call: earlier sessions for this model/reasoning must match design and inputs."""
    for version, path in existing_versions().items():
        info = json.loads(path.read_text(encoding="utf-8"))
        problems = []
        for k, v in DESIGN.items():
            earlier = info["design"].get(k, DESIGN_DEFAULTS_FOR_OLDER_RUNS.get(k))
            if earlier != v:
                problems.append(f"{k}: now {v!r}, earlier {earlier!r}")
        results = RESULTS_DIR / info["results_file"]
        if results.exists():
            with open(results, newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
            if header != RESULT_FIELDS:
                problems.append(f"{results.name} has different columns than this script writes "
                                f"(appending would misalign the data)")
        for k, h in hashes.items():
            if info["input_sha256"].get(k) != h:
                problems.append(f"{rel(INPUT_FILES[k])} has changed "
                                f"since the earlier session(s)")
        if problems:
            fail(problems + [f"(earlier sessions: {path.name}) Restore the inputs/settings, or "
                             f"change MODEL/REASONING_EFFORT to start a separate data file."],
                 "Settings or inputs differ from the existing data - no API calls were made")


def write_call_order(path, plan):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["call_order", "stimulus_id", "question_code", "repetition_index"])
        for p in plan:
            w.writerow([p["call_order"], p["stim"]["stimulus_id"], p["question_code"],
                        p["repetition_index"]])


def completed_rows(results_path):
    return lc.read_csv(results_path) if results_path.exists() else []


def item_key(item, version):
    return (item["stim"]["stimulus_id"], item["question_code"], str(item["repetition_index"]),
            MODEL, version, REASONING_EFFORT)


class RunInfo:
    """results/..._runinfo.json: parameters, input hashes, one entry per session, totals."""

    def __init__(self, path, version, hashes, results_path, adapter):
        self.path, self.results_path = path, results_path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {"model_requested": MODEL, "model_version": version,
                         "results_file": results_path.name, "design": DESIGN,
                         "input_sha256": hashes, "sessions": []}
        self.session = {"start_utc": lc.utc_now(), "end_utc": None, "status": "running",
                        "limits": LIMITS, "python_version": platform.python_version(),
                        "provider": PROVIDER, "sdk": adapter.sdk,
                        "sdk_version": lp.sdk_version(adapter),
                        "request_settings": {
                            **adapter.reasoning_request(REASONING_EFFORT),
                            "max_output_tokens": MAX_OUTPUT_TOKENS,
                            "temperature": "default" if TEMPERATURE is None else TEMPERATURE,
                            "image_detail": adapter.image_detail_label(IMAGE_DETAIL)},
                        "test_call": None, "calls_recorded": 0, "invalid_responses": 0,
                        "cost_recorded_usd": 0.0, "stop_reason": None}
        self.data["sessions"].append(self.session)

    def save(self):
        rows = completed_rows(self.results_path)
        tests = [s["test_call"]["total_cost"] for s in self.data["sessions"] if s.get("test_call")]
        self.data["totals"] = {
            "calls_completed": len(rows),
            "invalid_responses": sum(r["llm_invalid_response"] == "TRUE" for r in rows),
            "cost_recorded_calls_usd": round(sum(float(r["total_cost"]) for r in rows), 6),
            "cost_test_calls_usd": round(sum(float(c) for c in tests), 6),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        # Windows refuses the replace while another program (virus scanner, indexer, an editor)
        # briefly holds the file. Retry; never let this metadata file stop a paid run.
        for wait in (0.1, 0.3, 1.0, 2.0, None):
            try:
                tmp.replace(self.path)
                return
            except PermissionError:
                if wait is None:
                    print(f"    warning: could not update {self.path.name} (file locked); "
                          f"will retry after the next call")
                    return
                time.sleep(wait)


# =============================================================================
# Calls
# =============================================================================
def prices_at(when):
    """(input, output) price per 1M tokens for MODEL at a UTC time (peak pricing if any)."""
    m = MODELS[MODEL]
    peak = m.get("peak")
    if peak and not (peak.get("weekdays_only") and when.weekday() >= 5) and \
            any(start <= when.hour < end for start, end in peak["utc_hours"]):
        return peak["price_input"], peak["price_output"]
    return m["price_input"], m["price_output"]


def make_row(item, norm, latency, input_raw, n_frames, fps, image_detail_label):
    now = datetime.now(timezone.utc)
    scored = lc.score_response(norm, *prices_at(now))
    stim = item["stim"]
    return {
        **scored,
        "video_filename": item["video_filename"], "video_id": stim["video_id"],
        "noise_sd": stim["noise_sd"], "stimulus_id": stim["stimulus_id"],
        "question_code": item["question_code"], "question_text": item["question_text"],
        **item["meta"],
        "llm_model": MODEL, "temperature": "default" if TEMPERATURE is None else TEMPERATURE,
        "reasoning_setting": REASONING_EFFORT, "frames_sent": n_frames, "fps": fps,
        "image_detail": image_detail_label, "input_raw": input_raw,
        "datetime": now.isoformat(timespec="seconds"),
        "repetition_index": item["repetition_index"], "call_order": item["call_order"],
        "latency_sec": f"{latency:.2f}",
    }


def call(adapter, item, error_log):
    """Make one call. Returns (result row, provider-reported cost or None), or (None, None)
    if it failed after all retries."""
    parts, input_raw, n_frames, fps = lc.build_parts(item, PROJECT_DIR)
    req = adapter.build_request(MODEL, parts, reasoning_effort=REASONING_EFFORT,
                                max_output_tokens=MAX_OUTPUT_TOKENS, temperature=TEMPERATURE,
                                image_detail=IMAGE_DETAIL)
    resp, latency = lp.call_with_retries(adapter, req, max_retries=MAX_RETRIES,
                                         retry_waits=RETRY_WAITS_SEC, error_log=error_log)
    if resp is None:
        return None, None
    norm = adapter.normalize(resp)
    row = make_row(item, norm, latency, input_raw, n_frames, fps,
                   adapter.image_detail_label(IMAGE_DETAIL))
    return row, norm["provider_cost_usd"]


def images_processed(row):
    return int(row["tokens_input"]) >= int(row["frames_sent"]) * MIN_INPUT_TOKENS_PER_FRAME


def show_test_call(row, n_pending, n_plan, provider_cost):
    cost = float(row["total_cost"])
    print("\n--- Test call (not written to the results file) ------------------------")
    print(f"Stimulus {row['stimulus_id']}, question {row['question_code']}, "
          f"repetition {row['repetition_index']}")
    print(f"Model requested:   {MODEL}   (provider: {PROVIDER})")
    print(f"Model version:     {row['llm_model_version']}   (as returned by the API)")
    print(f"Response text:     {row['llm_answer_raw']!r}")
    print(f"Parsed answer:     {row['llm_answer'] or '-'}   invalid: {row['llm_invalid_response']}"
          f"   flags: {row['llm_answer_flag'] or '-'}")
    print(f"Finish status:     {row['finish_status']}   latency {row['latency_sec']} s")
    print(f"Tokens:            input {row['tokens_input']} (cached {row['tokens_input_cached']}), "
          f"output {row['tokens_output']} (reasoning {row['tokens_reasoning']})")
    print(f"Cost:              ${cost:.6f}   (computed from tokens x prices in MODELS)")
    if provider_cost is not None:
        print(f"Provider says:     ${provider_cost:.6f}")
        if cost > 0 and abs(provider_cost - cost) / cost > 0.25:
            print("  WARNING: the provider's cost differs from the computed cost by more than 25%. "
                  "Check the prices in MODELS and how reasoning tokens are counted before continuing.")
    print(f"Extrapolated:      ${cost * n_pending:.2f} for the {n_pending} remaining calls "
          f"(full design {n_plan} calls: ${cost * n_plan:.2f}); limit this session "
          f"${MAX_COST_USD:.2f}")
    print("-------------------------------------------------------------------------\n")


def confirm(args, reviewed_before):
    if args.yes and reviewed_before:
        print("Confirmed with --yes (a test call for this model version was made in an earlier session).")
        return True
    if args.yes:
        print("--yes is only accepted after a test call for this model version has been reviewed.")
    try:
        return input('Type "yes" to run the remaining calls: ').strip().lower() == "yes"
    except EOFError:
        print("No interactive input available. After reviewing the test call, re-run with --yes.")
        return False


# =============================================================================
# Summary
# =============================================================================
def summarize(results_path, questions, runinfo_path=None):
    rows = completed_rows(results_path)
    print(f"\n=== Summary: {results_path.name} ===")
    if not rows:
        print("No results yet.")
        return
    n_plan = len(VIDEOS) * len(NOISE_SDS) * len(questions) * N_REPS
    invalid = sum(r["llm_invalid_response"] == "TRUE" for r in rows)
    cost = sum(float(r["total_cost"]) for r in rows)
    test_cost = 0.0
    if runinfo_path and runinfo_path.exists():
        test_cost = json.loads(runinfo_path.read_text(encoding="utf-8")).get(
            "totals", {}).get("cost_test_calls_usd", 0.0)
    print(f"Calls completed: {len(rows)} of {n_plan}   invalid responses: {invalid}   "
          f"cost: ${cost:.2f} (+ ${test_cost:.2f} for unrecorded test calls)")
    flags = {}
    for r in rows:
        for f in filter(None, r["llm_answer_flag"].split(";")):
            flags[f] = flags.get(f, 0) + 1
    if flags:
        print("Flags: " + ", ".join(f"{k} {v}" for k, v in sorted(flags.items())))

    print("\nMean llm_answer (km/h) by question_code x noise_sd   [valid answers / calls]")
    print(f"{'question':<10}" + "".join(f"{'SD ' + str(sd):>16}" for sd in NOISE_SDS))
    for q in questions:
        code = q["question_code"]
        line = f"{code:<10}"
        for sd in NOISE_SDS:
            cell = [r for r in rows if r["question_code"] == code and int(r["noise_sd"]) == sd]
            vals = [float(r["llm_answer"]) for r in cell if r["llm_answer"] != ""]
            mean = f"{statistics.mean(vals):.1f}" if vals else "-"
            line += f"{mean + f' [{len(vals)}/{len(cell)}]':>16}"
        print(line)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Main data collection of the LLM L&P study.")
    parser.add_argument("--yes", action="store_true",
                        help="skip the prompt (only after an earlier test call for this version)")
    parser.add_argument("--check", action="store_true", help="input checks only, no API calls")
    parser.add_argument("--summary", action="store_true", help="summary only, no API calls")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)

    questions, meta, manifest = validate_inputs()
    plan = build_plan(questions, meta, manifest)
    hashes = {k: lc.sha256_file(p) for k, p in INPUT_FILES.items()}
    print(f"Design: {len(VIDEOS)} videos x {len(NOISE_SDS)} noise levels x {len(questions)} "
          f"questions x {N_REPS} reps = {len(plan)} calls. Model {MODEL}, reasoning "
          f"{REASONING_EFFORT}, image detail {IMAGE_DETAIL}, temperature "
          f"{'default' if TEMPERATURE is None else TEMPERATURE}.")

    if args.summary:
        versions = existing_versions()
        if not versions:
            print(f"No main results yet for {MODEL} / {REASONING_EFFORT}.")
        for version in versions:
            paths = output_paths(version)
            summarize(paths["results"], questions, paths["runinfo"])
        return

    check_identical_to_pilot(plan)
    check_previous_sessions(hashes)
    if args.check:
        print("All checks passed. No API calls were made (--check).")
        return

    try:
        from dotenv import load_dotenv
    except ImportError as e:
        sys.exit(f"Missing package ({e.name}). Run the script with the project's virtual "
                 f"environment:\n    .venv\\Scripts\\python run_main.py")
    load_dotenv(PROJECT_DIR / ".env")
    key_env = lp.PROVIDERS[PROVIDER]["key_env"]
    if not os.environ.get(key_env):
        sys.exit(f"{key_env} not found. Put it in .env as {key_env}=...")
    try:
        adapter = lp.make_adapter(PROVIDER, os.environ[key_env], REQUEST_TIMEOUT_SEC)
        available = adapter.list_models()
    except ImportError as e:
        sys.exit(f"Missing package ({e.name}) for provider {PROVIDER}. Install it into the "
                 f"project's virtual environment.")
    except Exception as e:
        sys.exit(f"Could not list the models of {PROVIDER} ({type(e).__name__}: {e}). "
                 f"Check {key_env} in .env.")
    if MODEL not in available:
        sys.exit(f"Model {MODEL!r} is not available to this {PROVIDER} API key.")
    error_log = RESULTS_DIR / f"{FILE_PREFIX}_{lc.safe_name(MODEL)}_{lc.safe_name(REASONING_EFFORT)}_errors.log"

    # ---- test call: checks image input and identifies the model version -----------------
    try:
        test, provider_cost = call(adapter, plan[0], error_log)
    except lp.FatalAPIError as e:
        sys.exit(f"Test call rejected by the API: {e}")
    if test is None:
        sys.exit("Test call failed after all retries; see the error log.")
    version = test["llm_model_version"]
    if not images_processed(test):
        sys.exit(f"Only {test['tokens_input']} input tokens for {test['frames_sent']} frames: "
                 f"{MODEL} does not seem to process the images. Stopping.")

    known = existing_versions()
    other = [v for v in known if v != version]
    if other:
        sys.exit(f"MODEL VERSION CHANGED: the API now returns {version!r}, but existing data for "
                 f"{MODEL} / {REASONING_EFFORT} is from {other}. Stopping without further calls; "
                 f"decide whether to continue under the new version.")

    paths = output_paths(version)
    reviewed_before = paths["runinfo"].exists()
    info = RunInfo(paths["runinfo"], version, hashes, paths["results"], adapter)
    info.session["test_call"] = {k: test[k] for k in (
        "stimulus_id", "question_code", "repetition_index", "llm_model_version", "llm_answer_raw",
        "llm_answer", "finish_status", "tokens_input", "tokens_input_cached", "tokens_output",
        "tokens_reasoning", "total_cost", "response_id", "latency_sec")}
    info.session["test_call"]["provider_reported_cost_usd"] = provider_cost
    write_call_order(paths["call_order"], plan)

    done = {tuple(r[k] for k in KEY_FIELDS) for r in completed_rows(paths["results"])}
    pending = [p for p in plan if item_key(p, version) not in done]
    calls, cost = 1, float(test["total_cost"])
    info.session["cost_session_usd"] = round(cost, 6)
    show_test_call(test, len(pending), len(plan), provider_cost)
    print(f"Results file: {rel(paths['results'])}  "
          f"({len(plan) - len(pending)} of {len(plan)} calls already done)")
    info.save()

    if not pending:
        info.session.update(status="finished", stop_reason="all calls already done",
                            end_utc=lc.utc_now())
        info.save()
        summarize(paths["results"], questions, paths["runinfo"])
        return
    if not confirm(args, reviewed_before):
        info.session.update(status="stopped", stop_reason="not confirmed", end_utc=lc.utc_now())
        info.save()
        print("Stopped before the main run.")
        return

    # ---- main loop -------------------------------------------------------------------------
    consecutive_failures, stop = 0, "all planned calls done"
    try:
        for k, item in enumerate(pending, start=1):
            if calls >= MAX_CALLS:
                stop = f"MAX_CALLS ({MAX_CALLS}) reached"
                break
            if cost > MAX_COST_USD:
                stop = f"cost ${cost:.2f} exceeds MAX_COST_USD ${MAX_COST_USD:.2f}"
                break
            label = (f"[{k}/{len(pending)}] #{item['call_order']} {item['stim']['stimulus_id']} "
                     f"{item['question_code']} rep {item['repetition_index']}")
            row, _ = call(adapter, item, error_log)
            if row is None:
                consecutive_failures += 1
                print(f"{label}: FAILED ({consecutive_failures} consecutive)")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    stop = f"{MAX_CONSECUTIVE_FAILURES} consecutive failed calls"
                    break
                continue
            consecutive_failures = 0
            calls += 1
            cost += float(row["total_cost"])
            info.session["cost_session_usd"] = round(cost, 6)

            if row["llm_model_version"] != version:
                lc.append_row(paths["mismatch"], RESULT_FIELDS, row)
                stop = (f"MODEL VERSION CHANGED during the run: {row['llm_model_version']!r} "
                        f"instead of {version!r} (call saved to {paths['mismatch'].name}, not to "
                        f"the results file)")
                break
            lc.append_row(paths["results"], RESULT_FIELDS, row)
            info.session["calls_recorded"] += 1
            info.session["invalid_responses"] += row["llm_invalid_response"] == "TRUE"
            info.session["cost_recorded_usd"] = round(
                info.session["cost_recorded_usd"] + float(row["total_cost"]), 6)
            info.save()

            note = f"  [{row['llm_answer_flag']}]" if row["llm_answer_flag"] else ""
            print(f"{label}: {row['llm_answer_raw'][:40]!r} -> {row['llm_answer'] or 'INVALID'}"
                  f"  ${float(row['total_cost']):.4f}  {row['latency_sec']} s{note}")
            if not images_processed(row):
                stop = (f"only {row['tokens_input']} input tokens for {row['frames_sent']} frames "
                        f"- images not processed")
                break
    except lp.FatalAPIError as e:
        stop = f"non-retryable API error: {e}"
    except KeyboardInterrupt:
        stop = "interrupted by user"
    finally:
        info.session.update(status="finished" if stop == "all planned calls done" else "stopped",
                            stop_reason=stop, end_utc=lc.utc_now())
        info.save()
        print(f"\nSession ended: {stop}. This session: {calls} calls incl. test call, ${cost:.2f}.")

    summarize(paths["results"], questions, paths["runinfo"])


if __name__ == "__main__":
    main()
