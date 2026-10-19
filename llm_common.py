"""
llm_common.py - shared code for the LLM Loftus & Palmer data collection.

Message construction, answer parsing and response scoring; the provider-specific API calls
are in llm_providers.py. The prompt texts and the message, parsing and scoring logic are taken
unchanged from run_pilot.py, which itself is left as it was (it documents how the pilot was
run). run_main.py checks at start-up that the messages are identical to the pilot's.
"""

import base64
import csv
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

# Prompt (identical for every call except the frames and the question) - same as the pilot
PROMPT_INTRO = ("The following images are frames from CCTV or dashcam footage capturing a car "
                "accident, sampled at {fps} frames per second ({n_frames} frames). Each frame is "
                "preceded by its timestamp.")
FRAME_LABEL = "t = {t:.2f} s"
PROMPT_QUESTION = ("Please answer the following question about the accident:\n{question_text}\n\n"
                   "The exact speed cannot be measured from the footage, so you should respond "
                   "with your best estimate. Answer with a single number representing speed in "
                   "km/h, and no other text.")


# =============================================================================
# Files
# =============================================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def append_row(path, fields, row):
    """Append one row (header first if the file is new) and force it to disk. Refuses to append
    to a file whose header differs from `fields`, which would misalign the columns."""
    new = not path.exists()
    if not new:
        with open(path, newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
        if header != list(fields):
            raise ValueError(f"{path.name}: existing columns differ from the columns being "
                             f"written; not appending.")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def append_log(path, message):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{utc_now()}  {message}\n")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_name(text):
    """Filesystem-safe version of a model/version/setting name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =============================================================================
# Stimuli and message construction
# =============================================================================
def source_video_filenames(raw_dir):
    """{video_id: source filename}, e.g. {"V1": "V1N.mp4"}."""
    out = {}
    for p in raw_dir.glob("*.mp4"):
        m = re.fullmatch(r"(V\d+)N?", p.stem)
        if m:
            out[m.group(1)] = p.name
    return out


def load_frames(project_dir, manifest_row):
    """Frame paths with timestamps, in time order, checked against the manifest."""
    frames_dir = project_dir / manifest_row["frames_dir"]
    frames = []
    for p in frames_dir.glob("*.png"):
        m = re.search(r"_t(\d+(?:\.\d+)?)\.png$", p.name)
        if m:
            frames.append((float(m.group(1)), p))
    frames.sort()
    if len(frames) != int(manifest_row["n_frames"]):
        sys.exit(f"{frames_dir}: found {len(frames)} frames, manifest says {manifest_row['n_frames']}")
    return frames


def image_b64(path):
    # Not cached: all frames of the main study would take ~300 MB of memory as base64.
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fmt_num(x):
    return f"{float(x):g}"


def build_parts(item, project_dir):
    """The message, provider-neutral: [{"type": "text", "text"} | {"type": "image", "path"}].
    Returns (parts, input_raw text with image paths, n_frames, fps)."""
    stim = item["stim"]
    frames = load_frames(project_dir, stim)
    fps = fmt_num(stim["target_fps"])
    intro = PROMPT_INTRO.format(fps=fps, n_frames=len(frames))
    question = PROMPT_QUESTION.format(question_text=item["question_text"])

    parts = [{"type": "text", "text": intro}]
    raw = [intro]
    for t, path in frames:
        label = FRAME_LABEL.format(t=t)
        parts += [{"type": "text", "text": label}, {"type": "image", "path": path}]
        raw += [label, f"[image: {path.relative_to(project_dir).as_posix()}]"]
    parts.append({"type": "text", "text": question})
    raw.append(question)
    return parts, "\n".join(raw), len(frames), fps


def openai_content(parts, image_detail):
    """Parts in the OpenAI Responses API format (as sent in the pilot)."""
    return [{"type": "input_text", "text": p["text"]} if p["type"] == "text" else
            {"type": "input_image", "detail": image_detail,
             "image_url": f"data:image/png;base64,{image_b64(p['path'])}"}
            for p in parts]


def build_message(item, project_dir, image_detail):
    """Returns (OpenAI content parts, input_raw text with image paths, n_frames, fps)."""
    parts, raw, n_frames, fps = build_parts(item, project_dir)
    return openai_content(parts, image_detail), raw, n_frames, fps


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
# Scoring a (normalized) response
# =============================================================================
def score_response(norm, price_input_per_1m, price_output_per_1m):
    """Answer, validity, flags, finish status, token usage and cost of one response.
    `norm` is a provider-independent response from llm_providers (same rules as the pilot)."""
    refusal, status = norm["refusal"], norm["status"]
    raw = norm["text"] if norm["text"] else (refusal or "")

    value, flags = parse_answer(raw)
    if refusal is not None:
        flags.append("refusal")
    if status != "completed":
        if status.startswith("incomplete"):
            flags.append("truncated")
        elif status != "refused":
            flags.append(f"status_{status}")
    invalid = value is None or status != "completed" or refusal is not None

    tok_in, tok_out = norm["tokens_input"], norm["tokens_output"]
    cost = (tok_in * price_input_per_1m + tok_out * price_output_per_1m) / 1e6
    return {
        "llm_answer_raw": raw,
        "llm_answer": "" if invalid else fmt_num(value),
        "llm_invalid_response": "TRUE" if invalid else "FALSE",
        "llm_answer_flag": ";".join(flags),
        "llm_model_version": norm["model_version"],
        "tokens_input": tok_in, "tokens_input_cached": norm["tokens_input_cached"],
        "tokens_output": tok_out,
        "tokens_reasoning": "" if norm["tokens_reasoning"] is None else norm["tokens_reasoning"],
        "total_cost": f"{cost:.6f}",
        "finish_status": status, "response_id": norm["response_id"],
    }
