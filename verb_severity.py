"""
verb_severity.py - order the question verbs by implied crash severity and speed using
semantic projection with text embeddings (Grand et al., 2022, Nature Human Behaviour).

For every embedding model x anchor set, an axis vector is built from anchor words/phrases
(mean of high-pole embeddings minus mean of low-pole embeddings). Each verb is embedded in
several sentence frames (and on its own, context-free) and projected onto that axis.

This script reads only config/ files and the embedding cache. It never reads experiment
results, so the ordering is independent of them.

Usage (from the project folder):
    .venv\\Scripts\\python verb_severity.py                     # all settings below
    .venv\\Scripts\\python verb_severity.py --yes               # skip the cost confirmation
    .venv\\Scripts\\python verb_severity.py --models text-embedding-3-small --anchor-sets S1
"""

import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# =============================================================================
# SETTINGS
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
ANCHORS_PATH = PROJECT_DIR / "config" / "anchors.csv"      # existing; read only
VERBS_PATH = PROJECT_DIR / "config" / "verbs.csv"
TEMPLATES_PATH = PROJECT_DIR / "config" / "templates.csv"
CACHE_PATH = PROJECT_DIR / "embeddings" / "cache.json"
RESULTS_DIR = PROJECT_DIR / "results"
EXCLUSIONS_PATH = RESULTS_DIR / "anchor_exclusions.csv"
LONG_PATH = RESULTS_DIR / "verb_severity_long.csv"
SUMMARY_PATH = RESULTS_DIR / "verb_severity_summary.csv"
ROBUSTNESS_PATH = RESULTS_DIR / "verb_severity_robustness.csv"
LOO_PATH = RESULTS_DIR / "verb_severity_leave_one_anchor_out.csv"
PLOT_PATH = RESULTS_DIR / "verb_severity_plot.png"

# Embedding models. Prices in USD per 1M tokens, verified 2026-10-10 on
# developers.openai.com/api/docs/models/text-embedding-3-{small,large}.
OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]
OPENAI_PRICES_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}
USE_LOCAL_MODEL = False                    # needs sentence-transformers + a ~400 MB download
LOCAL_MODEL = "all-mpnet-base-v2"
EMBED_BATCH_SIZE = 256
MAX_RETRIES = 3
RETRY_WAITS_SEC = [5, 15, 45]

# Anchor sets per axis (all sets in anchors.csv are validated; these are the ones used)
ANCHOR_SETS = ["S1", "S2", "S3", "V1", "V2"]
AXES = ["severity", "speed"]

# Verbs: the reference verb is scored and plotted, but kept out of z-score standardization,
# ranks, correlations, Kendall's W and the consensus order.
REFERENCE_VERBS = ["sc"]

# Preregistered anchor overlap filter: exclude an anchor if any of its words starts with a
# stem, either directly or after one of these prefixes (e.g. "untouched", "recontact").
OVERLAP_STEMS = ["touch", "contact", "hit", "bump", "collid", "collis", "crash", "smash"]
OVERLAP_PREFIXES = ["un", "re", "non", "pre", "mis", "over", "under", "out", "dis", "de", "co"]

# After a run with the full settings above (no --models/--anchor-sets overrides), write each
# verb's mean z-score per axis into these columns of verbs.csv (other columns are kept)
UPDATE_VERBS_CSV = True
VERBS_Z_COLUMNS = {"severity": "embedding_severity_z", "speed": "embedding_speed_z"}

# Context conditions: verb in sentence frames (score = mean over frames) vs verb phrase alone
CONTEXTS = ["templates", "context_free"]

# Plot colours
POINT_COLOR, REF_COLOR, INK, MUTED, SURFACE = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#fcfcfb"


# =============================================================================
# Inputs
# =============================================================================
def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("NA" if r.get(k) is None else r.get(k)) for k in fields})


def load_anchors():
    with open(ANCHORS_PATH, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = {"anchor_set", "axis", "pole", "text"} - set(reader.fieldnames or [])
        rows = list(reader)
    errors = []
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
    else:
        for i, r in enumerate(rows, start=2):
            for k in ("anchor_set", "axis", "pole", "text"):
                r[k] = (r[k] or "").strip()
            if r["axis"] not in AXES:
                errors.append(f"line {i}: axis {r['axis']!r} not in {AXES}")
            if r["pole"] not in ("low", "high"):
                errors.append(f"line {i}: pole {r['pole']!r} not in ['low', 'high']")
            if not r["text"]:
                errors.append(f"line {i}: empty text")
        for s in sorted({r["anchor_set"] for r in rows}):
            poles = {r["pole"] for r in rows if r["anchor_set"] == s}
            axes = {r["axis"] for r in rows if r["anchor_set"] == s}
            if poles != {"low", "high"}:
                errors.append(f"anchor set {s}: has poles {sorted(poles)}, needs low and high")
            if len(axes) != 1:
                errors.append(f"anchor set {s}: mixes axes {sorted(axes)}")
    if errors:
        sys.exit(f"{ANCHORS_PATH.name} failed validation:\n  " + "\n  ".join(errors))

    print(f"Anchors loaded from {ANCHORS_PATH.name}:")
    for s in sorted({r["anchor_set"] for r in rows}):
        rs = [r for r in rows if r["anchor_set"] == s]
        n_low = sum(r["pole"] == "low" for r in rs)
        print(f"  {s} ({rs[0]['axis']}): low {n_low}, high {len(rs) - n_low}")
    return rows


def overlap_reason(text):
    for word in re.findall(r"[a-z]+", text.lower()):
        for stem in OVERLAP_STEMS:
            if word.startswith(stem):
                return f"word '{word}' starts with stem '{stem}'"
            for pre in OVERLAP_PREFIXES:
                if word.startswith(pre + stem):
                    return f"word '{word}' contains stem '{stem}' after prefix '{pre}'"
    return None


def apply_overlap_filter(anchors):
    kept, excluded = [], []
    for a in anchors:
        reason = overlap_reason(a["text"])
        if reason:
            excluded.append({**a, "reason": reason})
        else:
            kept.append(a)
    write_csv(EXCLUSIONS_PATH, excluded, ["anchor_set", "axis", "pole", "text", "reason"])
    if excluded:
        print(f"Overlap filter: EXCLUDED {len(excluded)} anchor(s) (see {EXCLUSIONS_PATH.name}):")
        for e in excluded:
            print(f"  {e['anchor_set']} {e['pole']}: {e['text']!r} - {e['reason']}")
    else:
        print("Overlap filter: no anchors excluded.")
    for s in sorted({a["anchor_set"] for a in anchors}):
        if {a["pole"] for a in kept if a["anchor_set"] == s} != {"low", "high"}:
            sys.exit(f"Anchor set {s} lost a whole pole to the overlap filter.")
    return kept


def load_verbs_and_texts():
    """Returns verbs [(code, phrase)] and {(code, context): [(template_id, text)]}."""
    verbs = [(r["code"].strip(), r["verb_phrase"].strip()) for r in read_csv(VERBS_PATH)]
    templates = read_csv(TEMPLATES_PATH)
    texts = {}
    for code, phrase in verbs:
        if code in REFERENCE_VERBS:
            sents = [(t["template_id"], t["reference_sentence"]) for t in templates]
        else:
            sents = [(t["template_id"], t["template"].format(verb=phrase)) for t in templates]
        texts[(code, "templates")] = sents
        texts[(code, "context_free")] = [("none", phrase)]
    return verbs, texts


# =============================================================================
# Embeddings (cached)
# =============================================================================
def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    os.replace(tmp, CACHE_PATH)


def estimate_tokens(text):
    return math.ceil(len(text) / 3) + 1      # conservative; real counts are lower


def embed_openai(client, model, texts):
    import openai
    out = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
                if attempt == MAX_RETRIES:
                    sys.exit(f"Embedding request failed after {MAX_RETRIES + 1} attempts: {e}")
                wait = RETRY_WAITS_SEC[min(attempt, len(RETRY_WAITS_SEC) - 1)]
                print(f"  attempt {attempt + 1} failed ({type(e).__name__}); retrying in {wait} s")
                time.sleep(wait)
        out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        print(f"  {model}: embedded {len(batch)} texts ({resp.usage.total_tokens} tokens)")
    return out


def embed_local(texts):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        sys.exit("USE_LOCAL_MODEL needs sentence-transformers: "
                 ".venv\\Scripts\\python -m pip install sentence-transformers")
    return SentenceTransformer(LOCAL_MODEL).encode(texts).tolist()


def get_embeddings(models, texts, assume_yes):
    """{model: {text: unit vector}}, embedding only what is not cached yet."""
    cache = load_cache()
    todo = {m: [t for t in texts if t not in cache.get(m, {})] for m in models}
    paid = {m: ts for m, ts in todo.items() if ts and not m.startswith("local:")}

    if paid:
        print("\nTexts to embed through the OpenAI API (the rest are cached):")
        total = 0.0
        for m, ts in paid.items():
            tok = sum(estimate_tokens(t) for t in ts)
            cost = tok * OPENAI_PRICES_PER_1M[m] / 1e6
            total += cost
            print(f"  {m}: {len(ts)} texts, ~{tok} tokens, ~${cost:.6f}")
        print(f"  Estimated total: ~${total:.6f}")
        if not assume_yes:
            try:
                ok = input('Type "yes" to call the API: ').strip().lower() == "yes"
            except EOFError:
                ok = False
                print("No interactive input available; re-run with --yes to confirm.")
            if not ok:
                sys.exit("Stopped before calling the API.")

        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(PROJECT_DIR / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("OPENAI_API_KEY not found in .env")
        client = OpenAI(max_retries=0, timeout=60)
        for m, ts in paid.items():
            cache.setdefault(m, {}).update(zip(ts, embed_openai(client, m, ts)))
            save_cache(cache)

    for m, ts in todo.items():
        if ts and m.startswith("local:"):
            print(f"\nEmbedding {len(ts)} texts with local model {LOCAL_MODEL}")
            cache.setdefault(m, {}).update(zip(ts, embed_local(ts)))
            save_cache(cache)

    out = {}
    for m in models:
        out[m] = {}
        for t in texts:
            v = np.asarray(cache[m][t], dtype=np.float64)
            out[m][t] = v / np.linalg.norm(v)
    return out


# =============================================================================
# Semantic projection
# =============================================================================
def make_axis(low_vecs, high_vecs):
    """Returns a scoring function: 0 = midpoint, -0.5 = low-pole mean, +0.5 = high-pole mean."""
    mu_low, mu_high = np.mean(low_vecs, axis=0), np.mean(high_vecs, axis=0)
    a = mu_high - mu_low
    mid = (mu_high + mu_low) / 2
    aa = float(a @ a)
    return lambda e: float((e - mid) @ a) / aa


def verb_scores(score_fn, E, verbs, texts, context):
    """{code: (verb score, [(template_id, text, sentence score)])}"""
    out = {}
    for code, _ in verbs:
        sent = [(tid, t, score_fn(E[t])) for tid, t in texts[(code, context)]]
        out[code] = (float(np.mean([s for _, _, s in sent])), sent)
    return out


def ranks(values):
    """Average ranks (1 = lowest)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = np.array(ranks(x)), np.array(ranks(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def kendalls_w(rank_rows):
    """rank_rows: m specifications x n verbs, with tie correction."""
    R = np.asarray(rank_rows, dtype=float)
    m, n = R.shape
    if m < 2:
        return None
    S = float(((R.sum(axis=0) - m * (n + 1) / 2) ** 2).sum())
    T = 0.0
    for row in R:
        _, counts = np.unique(row, return_counts=True)
        T += float((counts ** 3 - counts).sum())
    return 12 * S / (m ** 2 * (n ** 3 - n) - m * T)


def short(model):
    return model.replace("text-embedding-", "").replace("local:", "")


def update_verbs_csv(consensus):
    """Write mean z-scores into verbs.csv, keeping all other columns and the row order."""
    with open(VERBS_PATH, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for axis, (_, mean_z, _) in consensus.items():
        col = VERBS_Z_COLUMNS[axis]
        if col not in fields:
            fields.append(col)
        for r in rows:
            r[col] = f"{round(mean_z[r['code'].strip()], 4):g}"
    tmp = VERBS_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    try:
        os.replace(tmp, VERBS_PATH)
    except PermissionError:
        tmp.unlink()
        print(f"\nCould not update {VERBS_PATH.name} (is it open in Excel?). Close it and re-run; "
              f"all embeddings are cached, so the re-run is free.")
        return
    cols = ", ".join(VERBS_Z_COLUMNS[a] for a in consensus)
    print(f"\nUpdated {cols} in {VERBS_PATH}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Order question verbs by semantic projection.")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    parser.add_argument("--models", nargs="+", help="override OPENAI_MODELS")
    parser.add_argument("--anchor-sets", nargs="+", help="override ANCHOR_SETS")
    args = parser.parse_args()
    openai_models = args.models or OPENAI_MODELS
    sets_used = args.anchor_sets or ANCHOR_SETS
    unknown = [m for m in openai_models if m not in OPENAI_PRICES_PER_1M]
    if unknown:
        sys.exit(f"No price set for {unknown}; add it to OPENAI_PRICES_PER_1M.")
    models = openai_models + ([f"local:{LOCAL_MODEL}"] if USE_LOCAL_MODEL else [])
    RESULTS_DIR.mkdir(exist_ok=True)

    anchors = apply_overlap_filter(load_anchors())
    missing_sets = [s for s in sets_used if s not in {a["anchor_set"] for a in anchors}]
    if missing_sets:
        sys.exit(f"Anchor set(s) {missing_sets} not in {ANCHORS_PATH.name}")
    anchors = [a for a in anchors if a["anchor_set"] in sets_used]
    set_axis = {a["anchor_set"]: a["axis"] for a in anchors}
    axes_used = [ax for ax in AXES if ax in set_axis.values()]

    verbs, texts = load_verbs_and_texts()
    ranked = [c for c, _ in verbs if c not in REFERENCE_VERBS]
    phrase = dict(verbs)
    all_texts = sorted({a["text"] for a in anchors} | {t for v in texts.values() for _, t in v})
    E = get_embeddings(models, all_texts, args.yes)

    # ---- scores for every specification (model x anchor set x context) -------------------
    long_rows, specs = [], {}       # specs[(model, set, context)] = {code: verb score}
    for model, s in itertools.product(models, sets_used):
        axis = set_axis[s]
        A = [a for a in anchors if a["anchor_set"] == s]
        fn = make_axis([E[model][a["text"]] for a in A if a["pole"] == "low"],
                       [E[model][a["text"]] for a in A if a["pole"] == "high"])
        for ctx in CONTEXTS:
            vs = verb_scores(fn, E[model], verbs, texts, ctx)
            specs[(model, s, ctx)] = {c: v for c, (v, _) in vs.items()}
            ref = [vs[c][0] for c in ranked]
            mu, sd = float(np.mean(ref)), float(np.std(ref, ddof=1))
            for code, (v, sents) in vs.items():
                for tid, text, sc in sents:
                    long_rows.append({
                        "model": model, "axis": axis, "anchor_set": s, "context": ctx,
                        "verb_code": code, "verb_phrase": phrase[code],
                        "is_reference": code in REFERENCE_VERBS, "template_id": tid, "text": text,
                        "score": round(sc, 6), "z_score": round((sc - mu) / sd, 6),
                        "verb_score": round(v, 6), "verb_z_score": round((v - mu) / sd, 6)})
    write_csv(LONG_PATH, long_rows, list(long_rows[0].keys()))

    def spec_name(k):
        return f"{short(k[0])}|{k[1]}|{k[2]}"

    # ---- summary and consensus per axis -------------------------------------------------
    summary_rows, robust_rows, loo_rows, consensus = [], [], [], {}
    for axis in axes_used:
        keys = [k for k in specs if set_axis[k[1]] == axis]
        z = {}
        for k in keys:
            vals = [specs[k][c] for c in ranked]
            mu, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
            z[k] = {c: (specs[k][c] - mu) / sd for c in specs[k]}
        rk = {k: dict(zip(ranked, ranks([specs[k][c] for c in ranked]))) for k in keys}
        mean_z = {c: float(np.mean([z[k][c] for k in keys])) for c in phrase}
        order = sorted(ranked, key=lambda c: mean_z[c])           # least -> most
        consensus[axis] = (order, mean_z, len(keys))
        for c in [*REFERENCE_VERBS, *ranked]:
            zs = [z[k][c] for k in keys]
            summary_rows.append({
                "axis": axis, "verb_code": c, "verb_phrase": phrase[c],
                "is_reference": c in REFERENCE_VERBS, "n_specifications": len(keys),
                "mean_z": round(mean_z[c], 4), "min_z": round(min(zs), 4), "max_z": round(max(zs), 4),
                "mean_rank": None if c in REFERENCE_VERBS else
                round(float(np.mean([rk[k][c] for k in keys])), 3),
                "consensus_position": None if c in REFERENCE_VERBS else order.index(c) + 1})

        # Spearman matrix and Kendall's W
        for ka, kb in itertools.product(keys, keys):
            robust_rows.append({
                "axis": axis, "measure": "spearman_rho", "subset": "all",
                "spec_a": spec_name(ka), "spec_b": spec_name(kb), "n_specs": 2,
                "n_verbs": len(ranked),
                "value": round(spearman([specs[ka][c] for c in ranked],
                                        [specs[kb][c] for c in ranked]), 4)})
        for subset in ["all", *CONTEXTS]:
            sub = [k for k in keys if subset in ("all", k[2])]
            w = kendalls_w([[rk[k][c] for c in ranked] for k in sub])
            robust_rows.append({
                "axis": axis, "measure": "kendalls_w", "subset": subset, "spec_a": None,
                "spec_b": None, "n_specs": len(sub), "n_verbs": len(ranked),
                "value": None if w is None else round(w, 4)})

        # Leave-one-anchor-out: does an adjacent consensus pair flip vs the full-anchor spec?
        pairs = list(zip(order, order[1:]))
        counts = {p: [0, 0, 0] for p in pairs}       # [fits, flips vs full spec, reversed vs consensus]
        for model, s in itertools.product(models, [s for s in sets_used if set_axis[s] == axis]):
            A = [a for a in anchors if a["anchor_set"] == s]
            for drop in range(len(A)):
                rest = A[:drop] + A[drop + 1:]
                fn = make_axis([E[model][a["text"]] for a in rest if a["pole"] == "low"],
                               [E[model][a["text"]] for a in rest if a["pole"] == "high"])
                for ctx in CONTEXTS:
                    full = specs[(model, s, ctx)]
                    loo = {c: v for c, (v, _) in verb_scores(fn, E[model], verbs, texts, ctx).items()}
                    for lo, hi in pairs:
                        counts[(lo, hi)][0] += 1
                        counts[(lo, hi)][1] += (loo[hi] > loo[lo]) != (full[hi] > full[lo])
                        counts[(lo, hi)][2] += loo[hi] < loo[lo]
        for (lo, hi), (n, flips, rev) in counts.items():
            loo_rows.append({"axis": axis, "lower_verb": lo, "higher_verb": hi,
                             "pair": f"{phrase[lo]} < {phrase[hi]}", "n_fits": n,
                             "swap_rate_vs_full_anchor_set": round(flips / n, 4),
                             "reversed_vs_consensus_rate": round(rev / n, 4)})

    write_csv(SUMMARY_PATH, summary_rows, list(summary_rows[0].keys()))
    write_csv(ROBUSTNESS_PATH, robust_rows, list(robust_rows[0].keys()))
    write_csv(LOO_PATH, loo_rows, list(loo_rows[0].keys()))

    # ---- printout -------------------------------------------------------------------------
    print(f"\nModels: {', '.join(models)}   Anchor sets: {', '.join(sets_used)}")
    for axis in axes_used:
        order, mean_z, m = consensus[axis]
        print(f"\n=== {axis.upper()}: consensus order, least -> most ({m} specifications) ===")
        for i, c in enumerate(order, 1):
            print(f"  {i}. {c} {phrase[c]:<14} mean z {mean_z[c]:+.2f}")
        for c in REFERENCE_VERBS:
            print(f"  (reference) {c} {phrase[c]:<14} mean z {mean_z[c]:+.2f}")
        for r in robust_rows:
            if r["axis"] == axis and r["measure"] == "kendalls_w":
                w = "n/a (needs 2+ specs)" if r["value"] is None else f"{r['value']:.3f}"
                print(f"  Kendall's W ({r['subset']}, {r['n_specs']} specs): {w}")
        keys = sorted({(r["spec_a"]) for r in robust_rows
                       if r["axis"] == axis and r["measure"] == "spearman_rho"})
        rho = {(r["spec_a"], r["spec_b"]): r["value"] for r in robust_rows
               if r["axis"] == axis and r["measure"] == "spearman_rho"}
        print("  Spearman rho between specifications:")
        for i, ka in enumerate(keys):
            print(f"    [{i + 1:>2}] {ka:<34}" + "".join(f"{rho[(ka, kb)]:>6.2f}" for kb in keys))
        print("  Leave-one-anchor-out, adjacent pairs (swap rate vs full anchor set):")
        for r in loo_rows:
            if r["axis"] == axis:
                print(f"    {r['pair']:<30} {r['swap_rate_vs_full_anchor_set']:>6.1%}  "
                      f"(reversed vs consensus {r['reversed_vs_consensus_rate']:.1%}, n={r['n_fits']})")

    plot(consensus, summary_rows, phrase, axes_used)
    print(f"\nWrote {', '.join(p.name for p in [LONG_PATH, SUMMARY_PATH, ROBUSTNESS_PATH, LOO_PATH, EXCLUSIONS_PATH, PLOT_PATH])}")

    if not UPDATE_VERBS_CSV:
        print(f"{VERBS_PATH.name} not updated (UPDATE_VERBS_CSV is off).")
    elif args.models or args.anchor_sets:
        print(f"{VERBS_PATH.name} not updated: --models/--anchor-sets were used (test settings).")
    else:
        update_verbs_csv(consensus)


def plot(consensus, summary_rows, phrase, axes_used):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(axes_used), figsize=(5.2 * len(axes_used), 4.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, axis in zip(axs[0], axes_used):
        order, _, m = consensus[axis]
        rows = {r["verb_code"]: r for r in summary_rows if r["axis"] == axis}
        ylist = [(c, i + 1) for i, c in enumerate(order)] + [(c, -0.2) for c in REFERENCE_VERBS]
        for c, y in ylist:
            r = rows[c]
            col = REF_COLOR if c in REFERENCE_VERBS else POINT_COLOR
            ax.hlines(y, r["min_z"], r["max_z"], color=col, linewidth=2, alpha=0.45)
            ax.scatter([r["mean_z"]], [y], s=46, color=col, edgecolors=SURFACE, linewidths=1.2,
                       zorder=3)
        ax.set_yticks([y for _, y in ylist],
                      [f"{c}  {phrase[c]}" + ("  (reference)" if c in REFERENCE_VERBS else "")
                       for c, _ in ylist])
        if REFERENCE_VERBS:
            ax.axhline(0.4, color="#c9c8c2", linewidth=0.8, linestyle=(0, (3, 3)))
        ax.axvline(0, color="#c9c8c2", linewidth=0.8)
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.tick_params(colors=MUTED, labelsize=9, left=False)
        ax.set_title(f"{axis.capitalize()} ({m} specifications)", loc="left", fontsize=10,
                     color=INK)
        ax.set_xlabel("z-score: point = mean, line = range across specifications",
                      fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
