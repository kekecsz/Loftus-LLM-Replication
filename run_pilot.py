"""
run_pilot.py - pilot of the LLM Loftus & Palmer replication through the OpenAI Responses API.

Each call sends one user message with the frames of one stimulus (from make_stimuli.py)
and one leading question, and asks for a speed estimate in km/h. Calls are independent
(no conversation history, store=False), run in a fixed-seed random order, and every
result is appended to the results CSV as soon as it arrives.

Usage (from the project folder):
    .venv\\Scripts\\python run_pilot.py            # test call, then asks you to type "yes"
    .venv\\Scripts\\python run_pilot.py --yes      # continue without the prompt (only once a
                                                  # test call is already in the results file)
    .venv\\Scripts\\python run_pilot.py --summary  # only print the summary and redraw the plot

Re-running is safe: every (stimulus, question, repetition, model, reasoning setting)
combination already in the results file is skipped, so paid calls are never repeated.
The API key is read from .env (OPENAI_API_KEY=...) and is never printed or logged.
"""

import argparse
import base64
import csv
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# PARAMETERS
# =============================================================================
RUN_NAME = "pilot2"                    # names the output files: results/{RUN_NAME}_results.csv

# Design
VIDEOS = ["V1", "V5"]                 # main study: ["V1", "V2", "V3", "V4", "V5"]
NOISE_SDS = [0, 80]                   # main study: [0, 25, 50, 80]
QUESTION_CODES = None                 # None = every question in config/questions.csv
N_REPS = 3                            # main study: 5
RANDOM_SEED = 20261010                # fixes the call order

# Model
MODEL = "gpt-6-luna"                  # a dated snapshot of it is used instead if OpenAI lists one
REASONING_EFFORT = "low"
TEMPERATURE = None                    # None = not sent (model default), logged as "default"
IMAGE_DETAIL = "high"                 # "low" would downscale the frames and smooth out the noise
MAX_OUTPUT_TOKENS = 4000              # includes hidden reasoning tokens

# Prices in USD per 1M tokens, verified 2026-10-10 on
# https://developers.openai.com/api/docs/models/gpt-6-luna (input $0.10, output $0.50).
# Cost is computed at the full input price; cached input ($0.01) makes the real bill lower.
PRICE_INPUT_PER_1M = 0.10
PRICE_OUTPUT_PER_1M = 0.50
MAIN_STUDY_N_CALLS = 5 * 4 * 8 * 5    # only used to extrapolate cost after the test call

# Safety limits (per run)
MAX_CALLS = 120
MAX_COST_USD = 5.0
MAX_RETRIES = 3                       # retries per call after the first attempt
RETRY_WAITS_SEC = [5, 15, 45]         # wait before retry 1, 2, 3
MAX_CONSECUTIVE_FAILURES = 5          # calls that failed after all retries
REQUEST_TIMEOUT_SEC = 180
MIN_INPUT_TOKENS_PER_FRAME = 50       # below this the images were evidently not processed

# Prompt (identical for every call except the frames and the question)
PROMPT_INTRO = ("The following images are frames from CCTV or dashcam footage capturing a car "
                "accident, sampled at {fps} frames per second ({n_frames} frames). Each frame is "
                "preceded by its timestamp.")
FRAME_LABEL = "t = {t:.2f} s"
PROMPT_QUESTION = ("Please answer the following question about the accident:\n{question_text}\n\n"
                   "The exact speed cannot be measured from the footage, so you should respond "
                   "with your best estimate. Answer with a single number representing speed in "
                   "km/h, and no other text.")

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "stimuli" / "manifest.csv"
QUESTIONS_PATH = PROJECT_DIR / "config" / "questions.csv"
RAW_VIDEO_DIR = PROJECT_DIR / "videos_raw"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / f"{RUN_NAME}_results.csv"
CALL_ORDER_PATH = RESULTS_DIR / f"{RUN_NAME}_call_order.csv"
ERROR_LOG_PATH = RESULTS_DIR / f"{RUN_NAME}_errors.log"
PLOT_PATH = RESULTS_DIR / f"{RUN_NAME}_plot.png"

RESULT_FIELDS = [
    "llm_answer_raw", "llm_answer", "llm_invalid_response", "llm_answer_flag",
    "video_filename", "video_id", "noise_sd", "stimulus_id", "question_code", "question_text",
    "llm_model", "llm_model_version", "temperature", "reasoning_setting",
    "tokens_input", "tokens_input_cached", "tokens_output", "tokens_reasoning", "total_cost",
    "frames_sent", "fps", "image_detail", "input_raw",
    "datetime", "repetition_index", "call_order", "finish_status", "response_id", "latency_sec",
]
KEY_FIELDS = ["stimulus_id", "question_code", "repetition_index", "llm_model", "reasoning_setting"]


# =============================================================================
# Inputs and design
# =============================================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def source_video_filenames():
    """{video_id: source filename}, e.g. {"V1": "V1N.mp4"}."""
    out = {}
    for p in RAW_VIDEO_DIR.glob("*.mp4"):
        m = re.fullmatch(r"(V\d+)N?", p.stem)
        if m:
            out[m.group(1)] = p.name
    return out


def load_frames(manifest_row):
    """Frame paths with timestamps, in time order, checked against the manifest."""
    frames_dir = PROJECT_DIR / manifest_row["frames_dir"]
    frames = []
    for p in frames_dir.glob("*.png"):
        m = re.search(r"_t(\d+(?:\.\d+)?)\.png$", p.name)
        if m:
            frames.append((float(m.group(1)), p))
    frames.sort()
    if len(frames) != int(manifest_row["n_frames"]):
        sys.exit(f"{frames_dir}: found {len(frames)} frames, manifest says {manifest_row['n_frames']}")
    return frames


def build_plan():
    manifest = {(r["video_id"], int(r["noise_sd"])): r for r in read_csv(MANIFEST_PATH)}
    questions = read_csv(QUESTIONS_PATH)
    if QUESTION_CODES is not None:
        questions = [q for q in questions if q["question_code"] in QUESTION_CODES]
    sources = source_video_filenames()

    plan = []
    for video in VIDEOS:
        for sd in NOISE_SDS:
            if (video, sd) not in manifest:
                sys.exit(f"No stimulus for {video} with noise SD {sd} in {MANIFEST_PATH.name}")
            stim = manifest[(video, sd)]
            for q in questions:
                for rep in range(1, N_REPS + 1):
                    plan.append({"stim": stim, "video_filename": sources.get(video, ""),
                                 "question_code": q["question_code"],
                                 "question_text": q["question_text"], "repetition_index": rep})
    random.Random(RANDOM_SEED).shuffle(plan)
    for i, p in enumerate(plan, start=1):
        p["call_order"] = i
    return plan


def write_call_order(plan):
    with open(CALL_ORDER_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["call_order", "stimulus_id", "question_code", "repetition_index"])
        for p in plan:
            w.writerow([p["call_order"], p["stim"]["stimulus_id"], p["question_code"],
                        p["repetition_index"]])


# =============================================================================
# Message construction
# =============================================================================
_b64_cache = {}


def image_b64(path):
    if path not in _b64_cache:
        _b64_cache[path] = base64.b64encode(path.read_bytes()).decode("ascii")
    return _b64_cache[path]


def fmt_num(x):
    return f"{float(x):g}"


def build_message(item):
    """Returns (content parts for the API, input_raw text with image paths, n_frames, fps)."""
    stim = item["stim"]
    frames = load_frames(stim)
    fps = fmt_num(stim["target_fps"])
    intro = PROMPT_INTRO.format(fps=fps, n_frames=len(frames))
    question = PROMPT_QUESTION.format(question_text=item["question_text"])

    content = [{"type": "input_text", "text": intro}]
    raw = [intro]
    for t, path in frames:
        label = FRAME_LABEL.format(t=t)
        content.append({"type": "input_text", "text": label})
        content.append({"type": "input_image", "detail": IMAGE_DETAIL,
                        "image_url": f"data:image/png;base64,{image_b64(path)}"})
        raw += [label, f"[image: {path.relative_to(PROJECT_DIR).as_posix()}]"]
    content.append({"type": "input_text", "text": question})
    raw.append(question)
    return content, "\n".join(raw), len(frames), fps


# =============================================================================
# Answer parsing
# =============================================================================
NUM = r"\d+(?:[.,]\d+)?"
RANGE_RE = re.compile(rf"({NUM})\s*(?:-|–|—|to)\s*({NUM})", re.IGNORECASE)


def to_float(s):
    return float(s.replace(",", "."))


def parse_answer(text):
    """Returns (value or None, list of flags). Accepts "45", "45 km/h", "~45"; a range -> midpoint."""
    t = text.strip()
    if not t:
        return None, ["empty"]
    flags = []
    m = RANGE_RE.search(t)
    if m:
        value = (to_float(m.group(1)) + to_float(m.group(2))) / 2
        flags.append("range_midpoint")
    else:
        nums = re.findall(NUM, t)
        if not nums:
            return None, ["no_number"]
        value = to_float(nums[0])
        if len(nums) > 1:
            flags.append("multiple_numbers")
    if re.search(r"\bmph\b|\bmiles?\b", t, re.I) and not re.search(r"km|kph|kilomet", t, re.I):
        flags.append("unit_mph")
    return value, flags


# =============================================================================
# API
# =============================================================================
class FatalAPIError(Exception):
    pass


def log_error(message):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}  {message}\n")


def resolve_model(client):
    """Use a dated snapshot of MODEL if OpenAI lists one; otherwise MODEL itself."""
    ids = {m.id for m in client.models.list()}
    snapshots = sorted(i for i in ids if re.fullmatch(re.escape(MODEL) + r"-\d{4}-\d{2}-\d{2}", i))
    if snapshots:
        print(f"Model: {MODEL} -> using dated snapshot {snapshots[-1]}")
        return snapshots[-1]
    if MODEL not in ids:
        sys.exit(f"Model {MODEL!r} is not available to this API key.")
    print(f"Model: {MODEL} (no dated snapshot listed)")
    return MODEL


def call_api(client, model, content):
    import openai
    kwargs = {"model": model, "input": [{"role": "user", "content": content}],
              "reasoning": {"effort": REASONING_EFFORT}, "max_output_tokens": MAX_OUTPUT_TOKENS,
              "store": False}
    if TEMPERATURE is not None:
        kwargs["temperature"] = TEMPERATURE

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.monotonic()
            resp = client.responses.create(**kwargs)
            return resp, time.monotonic() - t0
        except openai.APIStatusError as e:
            code = getattr(e, "code", None)
            transient = (e.status_code in (408, 409, 429) or e.status_code >= 500) \
                and code != "insufficient_quota"
            msg = f"HTTP {e.status_code} {code or ''}: {e.message}"
            if not transient:
                log_error(f"FATAL {msg}")
                raise FatalAPIError(msg) from None
        except openai.APIConnectionError as e:     # includes timeouts
            msg = f"{type(e).__name__}: {e}"
        log_error(f"attempt {attempt + 1} failed: {msg}")
        if attempt < MAX_RETRIES:
            wait = RETRY_WAITS_SEC[min(attempt, len(RETRY_WAITS_SEC) - 1)]
            print(f"    attempt {attempt + 1} failed ({msg}); retrying in {wait} s")
            time.sleep(wait)
    print(f"    failed after {MAX_RETRIES + 1} attempts ({msg})")
    return None, None


def response_to_row(resp, latency, item, model, input_raw, n_frames, fps):
    text, refusal = "", None
    for out in resp.output or []:
        if getattr(out, "type", None) == "message":
            for part in out.content or []:
                if part.type == "output_text":
                    text += part.text
                elif part.type == "refusal":
                    refusal = part.refusal
    raw = text if text else (refusal or "")

    status = resp.status or ""
    if status == "incomplete" and resp.incomplete_details is not None:
        status = f"incomplete:{resp.incomplete_details.reason}"

    value, flags = parse_answer(raw)
    if refusal is not None:
        flags.append("refusal")
    if status != "completed":
        flags.append("truncated" if status.startswith("incomplete") else f"status_{status}")
    invalid = value is None or status != "completed" or refusal is not None

    u = resp.usage
    tok_in = u.input_tokens if u else 0
    tok_out = u.output_tokens if u else 0
    tok_cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
    tok_reason = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
    cost = (tok_in * PRICE_INPUT_PER_1M + tok_out * PRICE_OUTPUT_PER_1M) / 1e6

    stim = item["stim"]
    return {
        "llm_answer_raw": raw,
        "llm_answer": "" if invalid else fmt_num(value),
        "llm_invalid_response": "TRUE" if invalid else "FALSE",
        "llm_answer_flag": ";".join(flags),
        "video_filename": item["video_filename"], "video_id": stim["video_id"],
        "noise_sd": stim["noise_sd"], "stimulus_id": stim["stimulus_id"],
        "question_code": item["question_code"], "question_text": item["question_text"],
        "llm_model": model, "llm_model_version": resp.model,
        "temperature": "default" if TEMPERATURE is None else TEMPERATURE,
        "reasoning_setting": REASONING_EFFORT,
        "tokens_input": tok_in, "tokens_input_cached": tok_cached, "tokens_output": tok_out,
        "tokens_reasoning": tok_reason, "total_cost": f"{cost:.6f}",
        "frames_sent": n_frames, "fps": fps, "image_detail": IMAGE_DETAIL, "input_raw": input_raw,
        "datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repetition_index": item["repetition_index"], "call_order": item["call_order"],
        "finish_status": status, "response_id": resp.id, "latency_sec": f"{latency:.2f}",
    }


def append_result(row):
    new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def run_one(client, model, item):
    content, input_raw, n_frames, fps = build_message(item)
    resp, latency = call_api(client, model, content)
    if resp is None:
        return None
    row = response_to_row(resp, latency, item, model, input_raw, n_frames, fps)
    append_result(row)
    if row["tokens_input"] < n_frames * MIN_INPUT_TOKENS_PER_FRAME:
        sys.exit(f"Only {row['tokens_input']} input tokens for {n_frames} frames: the model does "
                 f"not seem to process the images. Stopping.")
    return row


# =============================================================================
# Results, summary and plot
# =============================================================================
def item_key(item, model):
    return (item["stim"]["stimulus_id"], item["question_code"], str(item["repetition_index"]),
            model, REASONING_EFFORT)


def question_order():
    """Question codes in the row order of config/questions.csv (used for tables and plots)."""
    return [r["question_code"] for r in read_csv(QUESTIONS_PATH)
            if QUESTION_CODES is None or r["question_code"] in QUESTION_CODES]


def config_rows(model):
    if not RESULTS_PATH.exists():
        return []
    return [r for r in read_csv(RESULTS_PATH)
            if r["llm_model"] == model and r["reasoning_setting"] == REASONING_EFFORT]


def show_call(row, extrapolate_to=None):
    print("\n--- Test call ---------------------------------------------------------")
    print(f"Stimulus {row['stimulus_id']}, question {row['question_code']}, "
          f"repetition {row['repetition_index']} (call_order {row['call_order']})")
    print(f"Model returned:    {row['llm_model_version']}")
    print(f"Response text:     {row['llm_answer_raw']!r}")
    print(f"Parsed answer:     {row['llm_answer'] or '-'}   invalid: {row['llm_invalid_response']}"
          f"   flags: {row['llm_answer_flag'] or '-'}")
    print(f"Finish status:     {row['finish_status']}   latency {row['latency_sec']} s")
    print(f"Tokens:            input {row['tokens_input']} (cached {row['tokens_input_cached']}), "
          f"output {row['tokens_output']} (reasoning {row['tokens_reasoning']})")
    cost = float(row["total_cost"])
    print(f"Cost:              ${cost:.6f}")
    if extrapolate_to:
        print(f"Extrapolated:      ${cost * extrapolate_to:.4f} for this run's {extrapolate_to} calls,"
              f" ${cost * MAIN_STUDY_N_CALLS:.4f} for the main study ({MAIN_STUDY_N_CALLS} calls)")
    print("-----------------------------------------------------------------------\n")


def confirm(args, had_results_before):
    if args.yes and had_results_before:
        print('Confirmed with --yes.')
        return True
    if args.yes:
        print("--yes is only accepted once a test call has been reviewed; please confirm here.")
    try:
        return input('Type "yes" to run the remaining calls: ').strip().lower() == "yes"
    except EOFError:
        print("No interactive input available. After reviewing the test call, re-run with --yes.")
        return False


def summarize(model):
    rows = config_rows(model)
    if not rows:
        print("No results yet.")
        return
    invalid = [r for r in rows if r["llm_invalid_response"] == "TRUE"]
    flagged = [r for r in rows if r["llm_answer_flag"]]
    total = sum(float(r["total_cost"]) for r in rows)
    print(f"\n=== Summary: {model}, reasoning {REASONING_EFFORT} ===")
    print(f"Calls: {len(rows)}   invalid responses: {len(invalid)}   "
          f"flagged: {len(flagged)}   total cost: ${total:.4f}")
    versions = sorted({r["llm_model_version"] for r in rows})
    print(f"Model version(s) returned: {', '.join(versions)}")
    for r in flagged:
        print(f"  flag {r['llm_answer_flag']:<20} {r['stimulus_id']} {r['question_code']} "
              f"rep {r['repetition_index']}: {r['llm_answer_raw'][:60]!r}")

    cells = {}
    for r in rows:
        cells.setdefault((r["question_code"], r["video_id"], int(r["noise_sd"])), []).append(r)
    print(f"\n{'question':<9}{'video':<7}{'noise_sd':>8}{'n':>4}{'mean':>9}{'median':>9}  "
          f"{'identical':<11}answers")
    n_identical = n_complete = 0
    qpos = {c: i for i, c in enumerate(question_order())}
    for (q, v, sd) in sorted(cells, key=lambda k: (k[1], k[2], qpos.get(k[0], len(qpos)), k[0])):
        rs = sorted(cells[(q, v, sd)], key=lambda r: int(r["repetition_index"]))
        vals = [float(r["llm_answer"]) for r in rs if r["llm_answer"] != ""]
        complete = len(vals) == len(rs) == N_REPS
        same = complete and len(set(vals)) == 1
        n_complete += complete
        n_identical += same
        ident = "yes" if same else ("no" if complete else "incomplete")
        mean = f"{statistics.mean(vals):.1f}" if vals else "-"
        median = f"{statistics.median(vals):g}" if vals else "-"
        answers = ", ".join(r["llm_answer"] or "NA" for r in rs)
        print(f"{q:<9}{v:<7}{sd:>8}{len(rs):>4}{mean:>9}{median:>9}  {ident:<11}{answers}")
    print(f"\nRepetitions identical in {n_identical} of {n_complete} complete cells "
          f"({N_REPS} valid repetitions each).")


def plot(model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in config_rows(model) if r["llm_answer"] != ""]
    if not rows:
        return
    codes = question_order()
    xpos = {c: i for i, c in enumerate(codes)}
    point_color, mean_color, ink, muted = "#2a78d6", "#0b0b0b", "#0b0b0b", "#52514e"

    fig, axes = plt.subplots(len(VIDEOS), len(NOISE_SDS), sharex=True, sharey=True, squeeze=False,
                             figsize=(4.2 * len(NOISE_SDS) + 0.8, 3.0 * len(VIDEOS) + 0.9))
    fig.patch.set_facecolor("#fcfcfb")
    jitter = {1: [0.0]}
    for i, video in enumerate(VIDEOS):
        for j, sd in enumerate(NOISE_SDS):
            ax = axes[i][j]
            ax.set_facecolor("#fcfcfb")
            for code in codes:
                vals = [float(r["llm_answer"]) for r in rows if r["video_id"] == video
                        and int(r["noise_sd"]) == sd and r["question_code"] == code]
                if not vals:
                    continue
                n = len(vals)
                offs = jitter.get(n) or [(-0.12 + 0.24 * k / (n - 1)) for k in range(n)]
                x = xpos[code]
                ax.scatter([x + o for o in offs], vals, s=30, color=point_color, alpha=0.65,
                           edgecolors="#fcfcfb", linewidths=1, zorder=3)
                ax.hlines(statistics.mean(vals), x - 0.28, x + 0.28, color=mean_color,
                          linewidth=2, zorder=4)
            ax.set_title(f"{video} · noise SD {sd}", fontsize=10, color=ink, loc="left")
            ax.set_xticks(range(len(codes)), codes)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c9c8c2")
            ax.tick_params(colors=muted, labelsize=9)
            if j == 0:
                ax.set_ylabel("Speed estimate (km/h)", color=muted, fontsize=9)
    fig.suptitle(f"{model} · reasoning {REASONING_EFFORT} · dots = individual answers, "
                 f"bars = cell mean", fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {PLOT_PATH}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run the LLM Loftus & Palmer pilot.")
    parser.add_argument("--yes", action="store_true",
                        help="continue without the prompt (requires an earlier test call)")
    parser.add_argument("--summary", action="store_true",
                        help="only print the summary and redraw the plot")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        from openai import OpenAI
    except ImportError as e:
        sys.exit(f"Missing package ({e.name}). Run the script with the project's virtual "
                 f"environment:\n    .venv\\Scripts\\python run_pilot.py")

    load_dotenv(PROJECT_DIR / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=...")
    client = OpenAI(max_retries=0, timeout=REQUEST_TIMEOUT_SEC)   # retries handled here
    RESULTS_DIR.mkdir(exist_ok=True)

    model = resolve_model(client)
    if args.summary:
        summarize(model)
        plot(model)
        return

    plan = build_plan()
    write_call_order(plan)
    done = {tuple(r[k] for k in KEY_FIELDS) for r in config_rows(model)}
    pending = [p for p in plan if item_key(p, model) not in done]
    had_results_before = bool(done)
    print(f"Design: {len(plan)} calls ({len(VIDEOS)} videos x {len(NOISE_SDS)} noise levels x "
          f"{len(plan) // (len(VIDEOS) * len(NOISE_SDS) * N_REPS)} questions x {N_REPS} reps); "
          f"{len(plan) - len(pending)} already done, {len(pending)} to go. "
          f"Image detail: {IMAGE_DETAIL}, temperature: "
          f"{'default' if TEMPERATURE is None else TEMPERATURE}.")
    if not pending:
        summarize(model)
        plot(model)
        return

    calls = 0
    cost = 0.0
    try:
        if not had_results_before:
            # Test call = first call of the randomized order; it also checks image input.
            row = run_one(client, model, pending[0])
            if row is None:
                sys.exit("Test call failed; see the error log.")
            calls, cost = 1, float(row["total_cost"])
            pending = pending[1:]
            show_call(row, extrapolate_to=len(plan))
        else:
            prev = config_rows(model)
            per_call = sum(float(r["total_cost"]) for r in prev) / len(prev)
            print(f"Mean cost per completed call so far: ${per_call:.6f}; "
                  f"estimated cost of the remaining {len(pending)} calls: ${per_call * len(pending):.4f}")
        if not pending or not confirm(args, had_results_before):
            print("Stopped before the main run.")
            return

        consecutive_failures = 0
        for k, item in enumerate(pending, start=1):
            if calls >= MAX_CALLS:
                print(f"Stopping: MAX_CALLS ({MAX_CALLS}) reached for this run.")
                break
            if cost > MAX_COST_USD:
                print(f"Stopping: cost this run ${cost:.4f} exceeds MAX_COST_USD ${MAX_COST_USD}.")
                break
            label = (f"[{k}/{len(pending)}] #{item['call_order']} {item['stim']['stimulus_id']} "
                     f"{item['question_code']} rep {item['repetition_index']}")
            row = run_one(client, model, item)
            if row is None:
                consecutive_failures += 1
                print(f"{label}: FAILED ({consecutive_failures} consecutive)")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"Stopping: {MAX_CONSECUTIVE_FAILURES} consecutive failed calls.")
                    break
                continue
            consecutive_failures = 0
            calls += 1
            cost += float(row["total_cost"])
            note = f"  [{row['llm_answer_flag']}]" if row["llm_answer_flag"] else ""
            print(f"{label}: {row['llm_answer_raw'][:40]!r} -> {row['llm_answer'] or 'INVALID'}"
                  f"  ${float(row['total_cost']):.5f}  {row['latency_sec']} s{note}")
    except FatalAPIError as e:
        print(f"Stopping on a non-retryable API error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted. Completed calls are saved; re-run to continue.")
    finally:
        if calls:
            print(f"\nThis run: {calls} calls, ${cost:.4f}.")

    summarize(model)
    plot(model)


if __name__ == "__main__":
    main()
