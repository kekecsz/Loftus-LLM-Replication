"""
make_stimuli.py - prepare video stimuli for the LLM Loftus & Palmer replication.

For every source video and every noise level this script produces
  * a 3.4 s clip (1.9 s before to 1.5 s after first contact), 640x360, no audio,
    resampled to a constant VIDEO_FPS, with fresh Gaussian brightness noise per frame
  * PNG frames sampled at TARGET_FPS     -> stimuli/frames/{video}_sd{noise}/
  * a near-lossless H.264 MP4            -> stimuli/videos/{video}_sd{noise}.mp4
  * a contact sheet for visual checking  -> stimuli/contact_sheets/{video}_sd{noise}.png
  * a row in stimuli/manifest.csv

The PNG frames are a subset of the MP4's frames (same noise draw), so the
frame-based and native-video conditions show identical pixels.

Usage:
    python make_stimuli.py              # all videos in VIDEOS
    python make_stimuli.py V1N.mp4      # only the listed videos (manifest rows are merged)
"""

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "videos_raw"
IMPACT_CSV = PROJECT_DIR / "impact_times.csv"
OUT_DIR = PROJECT_DIR / "stimuli"
FRAMES_DIR = OUT_DIR / "frames"
VIDEOS_DIR = OUT_DIR / "videos"
CONTACT_DIR = OUT_DIR / "contact_sheets"
MANIFEST_PATH = OUT_DIR / "manifest.csv"

VIDEOS = ["V1N.mp4", "V2N.mp4", "V3N.mp4", "V4N.mp4", "V5N.mp4"]

# Clip window relative to the impact time (identical for every video)
PRE_IMPACT_SEC = 1.9
POST_IMPACT_SEC = 1.5
TIME_TOL_SEC = 1e-6          # tolerance for window bounds (allows a start of exactly 0)

# Output geometry / timing
OUT_WIDTH = 640
OUT_HEIGHT = 360
VIDEO_FPS = 30               # constant frame rate the clip is resampled to (MP4 rate)
TARGET_FPS = 6               # frame-sampling rate for the PNG frames

# Noise: Gaussian brightness noise (same value added to B, G and R), 0-255 scale
NOISE_SDS = [0, 25, 50, 80]
SEED_MULTIPLIER = 1000       # seed = video_number * SEED_MULTIPLIER + noise_sd

# MP4 encoding (H.264). CRF 1 = near-lossless with the widely supported "High" profile;
# CRF 0 would be lossless but forces "High 4:4:4 Predictive", which many decoders reject.
X264_CRF = 1
X264_PRESET = "slow"
PIX_FMT = "yuv420p"
COLOR_TAGS = ["-colorspace", "smpte170m", "-color_primaries", "smpte170m",
              "-color_trc", "smpte170m", "-color_range", "tv"]
# Exact BT.601 RGB<->YUV conversion; swscale's default fast path darkens by ~1 level
SWS_FLAGS = "accurate_rnd+full_chroma_int+bitexact"

# Frame filenames / contact sheets
TIMESTAMP_DECIMALS = 2
CONTACT_COLS = 7
CONTACT_TILE_SCALE = 0.5     # tiles downscaled with nearest-neighbour (keeps noise visible)
CONTACT_LABEL_H = 26
CONTACT_HEADER_H = 40
CONTACT_GAP = 6
CONTACT_IMPACT_COLOR = (0, 0, 255)   # BGR red border on the frame nearest to impact


# =============================================================================
# Helpers
# =============================================================================
def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        sys.exit("ffmpeg not found. Install it with:  python -m pip install imageio-ffmpeg")


def parse_decimal(text):
    """Parse '1.9' or '1,9' (and tolerate surrounding quotes/whitespace)."""
    return float(text.strip().strip('"').strip("'").strip().replace(",", "."))


def read_impact_times(path):
    """Return {filename: impact_time_sec}. Handles ',' or ';' delimiters and decimal commas."""
    lines = [ln for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    delim = ";" if ";" in lines[0] else ","
    header = [h.strip().strip('"').lower() for h in lines[0].split(delim)]
    if header[:2] != ["filename", "impact_time_sec"]:
        sys.exit(f"{path.name}: expected header 'filename,impact_time_sec', got {lines[0]!r}")
    times = {}
    for ln in lines[1:]:
        # split only at the first delimiter, so '1,9' survives in a comma-separated file
        name, value = ln.split(delim, 1)
        times[name.strip().strip('"')] = parse_decimal(value)
    return times


def video_id_and_number(filename):
    m = re.fullmatch(r"(V(\d+))N?", Path(filename).stem)
    if not m:
        sys.exit(f"Cannot derive a video ID from {filename!r} (expected e.g. V1N.mp4)")
    return m.group(1), int(m.group(2))


def frame_timestamps(ffmpeg, path):
    """Presentation timestamps (s) of every video frame, in decode/presentation order."""
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(path), "-map", "0:v:0",
           "-fps_mode", "passthrough", "-vf", "showinfo", "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if out.returncode != 0:
        sys.exit(f"ffmpeg failed reading timestamps of {path.name}:\n{out.stderr[-2000:]}")
    pts = [float(x) for x in re.findall(r"pts_time:\s*(-?[\d.]+)", out.stderr)]
    if not pts:
        sys.exit(f"No frame timestamps found for {path.name}")
    return np.array(pts)


def n_samples(duration, fps):
    """Number of timestamps k/fps in [0, duration)."""
    x = duration * fps
    return int(round(x)) if abs(x - round(x)) < 1e-6 else math.ceil(x)


def nearest_indices(times, targets):
    """Index into `times` of the element nearest to each target (ties -> earlier)."""
    return np.array([int(np.argmin(np.abs(times - t))) for t in targets])


def load_resampled_clip(ffmpeg, path, impact):
    """Trim, resample to VIDEO_FPS and scale. Returns (frames, clip_start, clip_end)."""
    clip_start = impact - PRE_IMPACT_SEC
    clip_end = impact + POST_IMPACT_SEC
    clip_len = PRE_IMPACT_SEC + POST_IMPACT_SEC

    pts = frame_timestamps(ffmpeg, path)
    frame_dur = float(np.median(np.diff(pts)[np.diff(pts) > 0]))
    video_end = pts[-1] + frame_dur
    if clip_start < pts[0] - TIME_TOL_SEC or clip_end > video_end + TIME_TOL_SEC:
        sys.exit(f"{path.name}: window {clip_start:.3f}-{clip_end:.3f} s (impact {impact} s) "
                 f"falls outside the video ({pts[0]:.3f}-{video_end:.3f} s). Not shortening it.")
    if abs(clip_start) < TIME_TOL_SEC:
        clip_start = 0.0

    out_times = clip_start + np.arange(n_samples(clip_len, VIDEO_FPS)) / VIDEO_FPS
    src_idx = nearest_indices(pts, out_times)
    worst = np.max(np.abs(pts[src_idx] - out_times))
    if worst > frame_dur:
        sys.exit(f"{path.name}: gap in source timestamps (nearest frame {worst:.3f} s away)")

    needed = set(src_idx.tolist())
    cap = cv2.VideoCapture(str(path))
    decoded = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in needed:
            if frame.shape[:2] != (OUT_HEIGHT, OUT_WIDTH):
                frame = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_AREA)
            decoded[i] = frame
        i += 1
    cap.release()
    if i != len(pts):
        sys.exit(f"{path.name}: OpenCV decoded {i} frames but ffmpeg reports {len(pts)}")

    frames = [decoded[j] for j in src_idx]
    return frames, clip_start, clip_end


def add_noise(frames, sd, seed):
    """Fresh Gaussian brightness noise for every frame; values rounded and clipped to 0-255."""
    if sd == 0:
        return [f.copy() for f in frames]
    rng = np.random.default_rng(seed)
    noisy = []
    for f in frames:
        noise = rng.standard_normal((OUT_HEIGHT, OUT_WIDTH), dtype=np.float32) * sd
        out = f.astype(np.float32) + noise[:, :, None]
        noisy.append(np.clip(np.rint(out), 0, 255).astype(np.uint8))
    return noisy


def write_mp4(ffmpeg, frames, path):
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_WIDTH}x{OUT_HEIGHT}",
           "-framerate", str(VIDEO_FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF),
           # format= must sit inside the filter chain: as an output option (-pix_fmt), together
           # with the colour tags, ffmpeg inserts a second, inexact converter after this one
           "-vf", f"scale=flags={SWS_FLAGS}:out_color_matrix=bt601:out_range=tv,format={PIX_FMT}",
           *COLOR_TAGS, "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(b"".join(f.tobytes() for f in frames))
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed writing {path.name}:\n{err.decode(errors='replace')}")


def check_mp4(ffmpeg, path, frames):
    """Decode the MP4 and compare it with the frames that went in. Returns (n, mean_abs, psnr).

    Decodes with ffmpeg's exact YUV->RGB conversion rather than cv2.VideoCapture, whose
    fast conversion adds its own ~1-level darkening that is not in the file.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-vf", f"scale=flags={SWS_FLAGS}:in_color_matrix=bt601:in_range=tv,format=bgr24",
           "-f", "rawvideo", "-"]
    out = subprocess.run(cmd, capture_output=True)
    if out.returncode != 0:
        sys.exit(f"ffmpeg failed decoding {path.name}:\n{out.stderr.decode(errors='replace')}")
    decoded = np.frombuffer(out.stdout, np.uint8).reshape(-1, OUT_HEIGHT, OUT_WIDTH, 3)
    if len(decoded) != len(frames):
        sys.exit(f"{path.name}: has {len(decoded)} frames, expected {len(frames)}")
    diff = decoded.astype(np.float32) - np.stack(frames).astype(np.float32)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    return len(decoded), float(np.mean(np.abs(diff))), psnr


def make_contact_sheet(frames, times, impact_in_clip, title, path):
    tile_w = round(OUT_WIDTH * CONTACT_TILE_SCALE)
    tile_h = round(OUT_HEIGHT * CONTACT_TILE_SCALE)
    cols = min(CONTACT_COLS, len(frames))
    rows = math.ceil(len(frames) / cols)
    cell_w, cell_h = tile_w + CONTACT_GAP, tile_h + CONTACT_LABEL_H + CONTACT_GAP
    sheet = np.full((CONTACT_HEADER_H + rows * cell_h + CONTACT_GAP,
                     cols * cell_w + CONTACT_GAP, 3), 255, np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, title, (CONTACT_GAP + 4, 27), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    impact_k = int(np.argmin(np.abs(np.asarray(times) - impact_in_clip)))

    for k, (frame, t) in enumerate(zip(frames, times)):
        x = CONTACT_GAP + (k % cols) * cell_w
        y = CONTACT_HEADER_H + (k // cols) * cell_h
        label = f"t={t:.{TIMESTAMP_DECIMALS}f}s"
        color = (0, 0, 0)
        if k == impact_k:
            label += f"  IMPACT ({impact_in_clip:.2f}s)"
            color = CONTACT_IMPACT_COLOR
        cv2.putText(sheet, label, (x + 2, y + 18), font, 0.5, color, 1, cv2.LINE_AA)
        ty = y + CONTACT_LABEL_H
        sheet[ty:ty + tile_h, x:x + tile_w] = cv2.resize(
            frame, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        if k == impact_k:
            cv2.rectangle(sheet, (x - 3, ty - 3), (x + tile_w + 2, ty + tile_h + 2),
                          CONTACT_IMPACT_COLOR, 3)
    cv2.imwrite(str(path), sheet)


def rel(path):
    return path.relative_to(PROJECT_DIR).as_posix()


def write_manifest(new_rows):
    """Merge new rows into an existing manifest (keyed by stimulus_id) and write it."""
    fields = ["stimulus_id", "video_id", "noise_sd", "seed", "target_fps", "n_frames",
              "clip_start_sec", "clip_end_sec", "impact_time_in_clip_sec",
              "frames_dir", "mp4_path"]
    rows = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, newline="", encoding="utf-8") as fh:
            rows = {r["stimulus_id"]: r for r in csv.DictReader(fh)}
    rows.update({r["stimulus_id"]: r for r in new_rows})
    ordered = sorted(rows.values(), key=lambda r: (int(r["video_id"][1:]), int(r["noise_sd"])))
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


# =============================================================================
# Main
# =============================================================================
def process_video(ffmpeg, filename, impact):
    video_id, number = video_id_and_number(filename)
    clip, clip_start, clip_end = load_resampled_clip(ffmpeg, RAW_DIR / filename, impact)
    impact_in_clip = round(impact - clip_start, 6)
    clip_len = PRE_IMPACT_SEC + POST_IMPACT_SEC

    # PNG frames: the VIDEO_FPS frame nearest to each TARGET_FPS timestamp
    png_times = np.arange(n_samples(clip_len, TARGET_FPS)) / TARGET_FPS
    png_idx = nearest_indices(np.arange(len(clip)) / VIDEO_FPS, png_times)

    rows = []
    for sd in NOISE_SDS:
        stim_id = f"{video_id}_sd{sd}"
        seed = number * SEED_MULTIPLIER + sd
        noisy = add_noise(clip, sd, seed)

        frames_dir = FRAMES_DIR / stim_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old in frames_dir.glob("*.png"):
            old.unlink()
        sampled = [noisy[j] for j in png_idx]
        for frame, t in zip(sampled, png_times):
            cv2.imwrite(str(frames_dir / f"{stim_id}_t{t:.{TIMESTAMP_DECIMALS}f}.png"), frame)

        mp4_path = VIDEOS_DIR / f"{stim_id}.mp4"
        write_mp4(ffmpeg, noisy, mp4_path)
        n_mp4, mad, psnr = check_mp4(ffmpeg, mp4_path, noisy)

        title = (f"{stim_id}   noise SD {sd}   seed {seed}   {TARGET_FPS} fps   "
                 f"source {clip_start:.2f}-{clip_end:.2f} s")
        make_contact_sheet(sampled, png_times, impact_in_clip, title,
                           CONTACT_DIR / f"{stim_id}.png")

        print(f"  {stim_id}: seed {seed}, {len(sampled)} PNGs, MP4 {n_mp4} frames @ {VIDEO_FPS} fps, "
              f"{mp4_path.stat().st_size / 1e6:.1f} MB, MP4 vs frames: mean |diff| {mad:.3f}, "
              f"PSNR {psnr:.1f} dB")
        rows.append({
            "stimulus_id": stim_id, "video_id": video_id, "noise_sd": sd, "seed": seed,
            "target_fps": TARGET_FPS, "n_frames": len(sampled),
            "clip_start_sec": f"{clip_start:.3f}", "clip_end_sec": f"{clip_end:.3f}",
            "impact_time_in_clip_sec": f"{impact_in_clip:.3f}",
            "frames_dir": rel(frames_dir), "mp4_path": rel(mp4_path),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("videos", nargs="*", default=VIDEOS,
                        help="source filenames to process (default: all in VIDEOS)")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    impact_times = read_impact_times(IMPACT_CSV)
    if TARGET_FPS > VIDEO_FPS:
        sys.exit("TARGET_FPS must not exceed VIDEO_FPS")
    for d in (FRAMES_DIR, VIDEOS_DIR, CONTACT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rows = []
    for filename in args.videos:
        if filename not in impact_times:
            sys.exit(f"{filename} not listed in {IMPACT_CSV.name}")
        print(f"{filename} (impact {impact_times[filename]} s)")
        rows += process_video(ffmpeg, filename, impact_times[filename])
    write_manifest(rows)
    print(f"Wrote {len(rows)} rows to {rel(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
