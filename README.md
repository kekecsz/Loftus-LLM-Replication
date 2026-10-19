# Loftus & Palmer (1974) replication with multimodal LLMs

Do leading questions bias speed estimates by large language models the way they bias human
eyewitnesses? Models are shown frames from real car-crash footage and asked how fast the car
was going "right before it *touched* / *bumped into* / … / *smashed into* the other car",
with a neutral control ("right before the two cars' paths met"). Visual uncertainty is
manipulated by adding Gaussian pixel noise to the footage.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Stimuli | `make_stimuli.py` | `stimuli/`: 5 videos × 4 noise levels (SD 0, 25, 50, 80) |
| 2. Pilot | `run_pilot.py` | `results/pilot_*` |
| 3. Verb ordering | `verb_severity.py` | `results/verb_severity_*`, z-scores in `config/verbs.csv` |
| 4. Main data collection | `run_main.py` (+ `llm_common.py`, `llm_providers.py`) | `results/main_*` |

**1. Stimuli.** Each source video (`videos_raw/`) is trimmed to a 3.4 s window from 1.9 s before
to 1.5 s after first contact (`impact_times.csv`), scaled to 640×360, resampled to a constant
30 fps and degraded with Gaussian brightness noise, freshly drawn for every frame
(seed = video number × 1000 + noise SD). Frames are sampled at 6 fps: 21 PNG frames per
stimulus, t = 0.00–3.33 s. The script also writes near-lossless H.264 MP4s, contact sheets
(`stimuli/contact_sheets/`) and `stimuli/manifest.csv`.

**2. Pilot.** `gpt-6-luna`, videos V1 and V5 × noise SD 0 and 80 × 8 questions × 3 repetitions
(96 calls).

**3. Verb ordering.** The question verbs are ordered by implied crash severity and speed with
semantic projection on text embeddings (Grand et al., 2022): two OpenAI embedding models,
anchor sets in `config/anchors.csv`, verbs in sentence frames (`config/templates.csv`) and on
their own. The script never reads experiment results. The mean z-scores are written to
`config/verbs.csv` (`embedding_severity_z`, `embedding_speed_z`).

**4. Main data collection.** Fully crossed: 5 videos × 4 noise levels × 8 questions
(`config/questions.csv`) × 5 repetitions = 800 calls per model, in a fixed-seed random order.
Every call is independent: one user message with an intro, 21 time-stamped frames and the
question, asking for a single number in km/h. The message content is identical for all
models and identical to the pilot (checked at start-up); only the request format differs
between providers. Questions s1–s5 form the confirmatory set (`confirmatory_set`,
`verb_rank` 1–5); `sc` (neutral control), `sx` (hit) and `sy` (crashed into) are not ranked.

## Data collected

| Model | Provider | Version returned by the API | Dates (UTC) | Calls | Invalid |
|---|---|---|---|---|---|
| gpt-6.1-sol | OpenAI | gpt-6.1-sol | 2026-10-11 | 800 | 0 |
| gpt-6-luna | OpenAI | gpt-6-luna | 2026-10-11 | 800 | 0 |
| grok-4.7 | xAI | grok-4.7 | 2026-10-11 – 10-12 | 800 | 0 |
| claude-haiku-5-5 | Anthropic | claude-haiku-5-5 | 2026-10-12 | 800 | 0 |
| gemini-3.5-flash-lite | Google | gemini-3.5-flash-lite | 2026-10-12 | 800 | 0 |
| deepseek-flash (V4.1 Flash) | DeepSeek | deepseek-flash | 2026-10-12 | 800 | 354 |

All runs used reasoning effort "low", no temperature setting and a 4,000-token output limit.
The 354 invalid DeepSeek responses all reached the output limit while reasoning, before any
answer text (`finish_status` = `incomplete:max_output_tokens`).

Each `results/main_{model}_{version}_{reasoning}.csv` has one row per call. Main columns:
`llm_answer` (parsed km/h; empty if invalid), `llm_answer_raw`, `llm_invalid_response`,
`llm_answer_flag`, `video_id`, `noise_sd`, `question_code`, `verb_phrase`,
`embedding_severity_z`, `embedding_speed_z`, `confirmatory_set`, `verb_rank`,
`repetition_index`, `call_order`, token counts and cost, `finish_status`, `input_raw` (the full
message, with image file paths in place of the images). The matching `_runinfo.json` records
all settings, the provider and SDK versions, the test call of each session and SHA-256 hashes
of the input files; `_call_order.csv` is the planned call order.

## Notes for using the data

- **Question codes changed after the pilot.** `results/pilot_results.csv` and the
  `results/verb_severity_*` files use the old codes; join them to later data on
  `question_text` or `verb_phrase`, never on the code.

  | Verb | Old code (pilot, verb_severity outputs) | Current code |
  |---|---|---|
  | paths met (control) | s0 | sc |
  | touched | s1 | s1 |
  | bumped into | s4 | s2 |
  | contacted | s2 | s3 |
  | collided with | s5 | s4 |
  | smashed into | s7 | s5 |
  | hit | s3 | sx |
  | crashed into | s6 | sy |

- **"Low" reasoning is not the same setting everywhere.** OpenAI and xAI: `reasoning.effort`;
  Anthropic: adaptive thinking with `output_config.effort`; Google: `thinking_level` LOW;
  DeepSeek: `reasoning_effort` with thinking on. The exact request settings are in each
  `_runinfo.json` (`request_settings`).
- **Image handling differs by provider.** The same 640×360 PNGs are sent to every model:
  OpenAI and xAI with detail "high", DeepSeek with detail "high" (original size), Google with
  media resolution HIGH, Anthropic without a resolution setting (`image_detail` column).
- **Model versions.** Some providers return an alias rather than a dated snapshot as the
  version (e.g. `deepseek-flash`, which DeepSeek maps to its current Flash model), so the
  run dates are part of the model identification.
- **Costs** are computed from token counts × list prices at full input price; cached input
  makes the actual bill lower.

## Reproducing

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`make_stimuli.py` additionally needs `pip install -r requirements-stimuli.txt` (opencv,
ffmpeg via imageio-ffmpeg). Use exactly these versions to regenerate identical stimuli: the
frames and MP4s in `stimuli/frames/` and `stimuli/videos/` are not in the repository and are
rebuilt by the script from `videos_raw/`.

API keys go in a `.env` file in the project folder (never committed); only the keys of the
providers you run are needed:

```
OPENAI_API_KEY=...
XAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
```

Set `MODEL` at the top of `run_main.py` (one of the entries in `MODELS`) and run
`.venv\Scripts\python run_main.py`. It first checks the inputs, makes one unrecorded test
call and asks for confirmation before the paid run; `--check` runs the checks without any API
call, `--summary` summarizes existing results. Re-running continues an interrupted run
without repeating completed calls. All parameters are at the top of each script.

Data files are stored byte-for-byte (`.gitattributes`), so the input-file hashes in the
`_runinfo.json` files match after cloning.

## Video sources

See `videos_raw/raw_video_sources.txt`. V1–V4: compilation of dashcam/CCTV car-crash videos
(https://www.kaggle.com/datasets/fahaddalwai/cctvfootagevideo); V5: dashcam video of a T-bone
crash (https://www.youtube.com/watch?v=FD1sacdeW8E).

## Reference

Loftus, E. F., & Palmer, J. C. (1974). Reconstruction of automobile destruction: An example of
the interaction between language and memory. *Journal of Verbal Learning and Verbal Behavior,
13*(5), 585–589.

Grand, G., Blank, I. A., Pereira, F., & Fedorenko, E. (2022). Semantic projection recovers rich
human knowledge of multiple object features from word embeddings. *Nature Human Behaviour, 6*,
975–987.
