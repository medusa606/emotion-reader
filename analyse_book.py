#!/usr/bin/env python3
"""
Full-book emotion analyser.
Processes every sentence in a Gutenberg book, records per-sentence
emotion probabilities + VA scores, prints performance stats and saves
results to CSV/JSON + a sentiment timeline plot.

Usage:
    python analyse_book.py --book 345          # Dracula (default)
    python analyse_book.py --book 84 --batch 32
    python analyse_book.py --list              # show available books
"""

import os
import re
import sys
import json
import time
import argparse
import threading
import warnings
from datetime import datetime
from pathlib import Path

import requests
import nltk
import torch
import psutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from transformers import pipeline

# Suppress tokeniser parallelism warning when using batches
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)

# ── NLTK ──────────────────────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# BOOK CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
BOOKS = {
    "1342": {"title": "Pride and Prejudice",          "author": "Jane Austen"},
    "2701": {"title": "Moby Dick",                    "author": "Herman Melville"},
    "174":  {"title": "The Picture of Dorian Gray",   "author": "Oscar Wilde"},
    "1661": {"title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle"},
    "84":   {"title": "Frankenstein",                 "author": "Mary Shelley"},
    "345":  {"title": "Dracula",                      "author": "Bram Stoker"},
    "1400": {"title": "Great Expectations",           "author": "Charles Dickens"},
}

GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION METADATA  (VA scores from Russell's circumplex)
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_META = {
    "fear":           {"group": "Tension / Danger",     "colour": "#c0392b", "v": -0.72, "a":  0.82},
    "nervousness":    {"group": "Tension / Danger",     "colour": "#e74c3c", "v": -0.50, "a":  0.68},
    "anger":          {"group": "Tension / Danger",     "colour": "#e67e22", "v": -0.60, "a":  0.82},
    "annoyance":      {"group": "Tension / Danger",     "colour": "#f39c12", "v": -0.38, "a":  0.50},
    "disgust":        {"group": "Drama / Darkness",     "colour": "#8e44ad", "v": -0.68, "a":  0.40},
    "grief":          {"group": "Drama / Darkness",     "colour": "#6c3483", "v": -0.82, "a": -0.30},
    "sadness":        {"group": "Drama / Darkness",     "colour": "#5b2c6f", "v": -0.72, "a": -0.25},
    "remorse":        {"group": "Drama / Darkness",     "colour": "#76448a", "v": -0.62, "a": -0.20},
    "joy":            {"group": "Positive / Uplifting", "colour": "#27ae60", "v":  0.82, "a":  0.72},
    "love":           {"group": "Positive / Uplifting", "colour": "#e91e63", "v":  0.80, "a":  0.52},
    "admiration":     {"group": "Positive / Uplifting", "colour": "#2ecc71", "v":  0.62, "a":  0.42},
    "excitement":     {"group": "Positive / Uplifting", "colour": "#f1c40f", "v":  0.72, "a":  0.90},
    "amusement":      {"group": "Positive / Uplifting", "colour": "#1abc9c", "v":  0.70, "a":  0.60},
    "gratitude":      {"group": "Positive / Uplifting", "colour": "#16a085", "v":  0.72, "a":  0.38},
    "pride":          {"group": "Positive / Uplifting", "colour": "#148f77", "v":  0.70, "a":  0.50},
    "optimism":       {"group": "Positive / Uplifting", "colour": "#45b39d", "v":  0.70, "a":  0.48},
    "relief":         {"group": "Positive / Uplifting", "colour": "#52be80", "v":  0.52, "a": -0.15},
    "surprise":       {"group": "Contemplative",        "colour": "#3498db", "v":  0.10, "a":  0.72},
    "curiosity":      {"group": "Contemplative",        "colour": "#2980b9", "v":  0.30, "a":  0.52},
    "realization":    {"group": "Contemplative",        "colour": "#1f618d", "v":  0.12, "a":  0.40},
    "confusion":      {"group": "Contemplative",        "colour": "#5d6d7e", "v": -0.22, "a":  0.42},
    "approval":       {"group": "Contemplative",        "colour": "#85c1e9", "v":  0.50, "a":  0.28},
    "desire":         {"group": "Contemplative",        "colour": "#a9cce3", "v":  0.40, "a":  0.62},
    "neutral":        {"group": "Neutral / Flat",       "colour": "#95a5a6", "v":  0.00, "a": -0.40},
    "caring":         {"group": "Neutral / Flat",       "colour": "#abebc6", "v":  0.58, "a": -0.10},
    "disapproval":    {"group": "Neutral / Flat",       "colour": "#aab7b8", "v": -0.48, "a":  0.38},
    "embarrassment":  {"group": "Neutral / Flat",       "colour": "#d7bde2", "v": -0.38, "a":  0.50},
    "disappointment": {"group": "Neutral / Flat",       "colour": "#a2d9ce", "v": -0.58, "a": -0.20},
}

ALL_LABELS = list(EMOTION_META.keys())

# Groups shown on the timeline plot (neutral excluded)
TIMELINE_GROUPS = {
    "Tension / Danger":    "#e74c3c",
    "Drama / Darkness":    "#8e44ad",
    "Positive / Uplifting":"#27ae60",
    "Contemplative":       "#3498db",
}

# Hartmann → GoEmotions mapping for ensemble blend
HARTMANN_MAP = {
    "anger":    ["anger", "annoyance"],
    "disgust":  ["disgust", "disapproval"],
    "fear":     ["fear", "nervousness"],
    "joy":      ["joy", "excitement", "amusement"],
    "neutral":  ["neutral"],
    "sadness":  ["sadness", "grief", "remorse"],
    "surprise": ["surprise", "realization"],
}

HIGH_AROUSAL_LABELS = {
    "fear", "anger", "nervousness", "excitement",
    "disgust", "grief", "sadness", "surprise",
}

# Lines per pseudo-page (matches the Flask app)
LINES_PER_PAGE = 40


# ─────────────────────────────────────────────────────────────────────────────
# GUTENBERG HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_text(book_id: str) -> str:
    url  = GUTENBERG_URL.format(id=book_id)
    print(f"  Fetching {url} …")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    try:
        raw = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        raw = resp.content.decode("latin-1")
    return raw


def strip_boilerplate(text: str) -> str:
    starts = ["*** START OF THE PROJECT GUTENBERG",
              "***START OF THE PROJECT GUTENBERG",
              "*** START OF THIS PROJECT GUTENBERG"]
    ends   = ["*** END OF THE PROJECT GUTENBERG",
              "***END OF THE PROJECT GUTENBERG",
              "*** END OF THIS PROJECT GUTENBERG"]
    s = 0
    for m in starts:
        i = text.find(m)
        if i != -1:
            s = text.index("\n", i) + 1
            break
    e = len(text)
    for m in ends:
        i = text.find(m)
        if i != -1:
            e = i
            break
    return text[s:e].strip()


def assign_page_numbers(sentences: list[str], full_text: str) -> list[int]:
    """
    Give each sentence a page number based on which line block it falls in.
    We locate each sentence in the original line-split text.
    """
    lines = [l for l in full_text.splitlines() if l.strip()]
    # Build a char-offset map: line index → page
    page_of_line = [i // LINES_PER_PAGE + 1 for i in range(len(lines))]

    # Join lines for substring search
    joined = "\n".join(lines)

    pages = []
    search_start = 0
    for sent in sentences:
        # Find first occurrence from where we left off
        needle = sent[:60].strip()   # use first 60 chars to locate
        idx    = joined.find(needle, search_start)
        if idx == -1:
            # fallback — count newlines up to last found position
            pages.append(pages[-1] if pages else 1)
            continue
        line_no = joined[:idx].count("\n")
        page_no = line_no // LINES_PER_PAGE + 1
        pages.append(page_no)
        search_start = idx
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# GPU MONITOR (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

class ResourceMonitor:
    """Samples CPU and GPU utilisation at ~1 Hz in a daemon thread."""

    def __init__(self):
        self.cpu_samples  = []
        self.gpu_samples  = []
        self.vram_samples = []
        self._stop        = threading.Event()
        self._thread      = threading.Thread(target=self._run, daemon=True)
        self._has_gpu     = torch.cuda.is_available()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _run(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            if self._has_gpu:
                try:
                    # Use torch directly — avoids needing GPUtil
                    mem   = torch.cuda.memory_stats(0)
                    alloc = mem.get("allocated_bytes.all.current", 0) / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    util  = alloc / total * 100
                    self.gpu_samples.append(util)
                    self.vram_samples.append(alloc)
                except Exception:
                    pass
            self._stop.wait(1.0)

    @property
    def avg_cpu(self):
        return float(np.mean(self.cpu_samples)) if self.cpu_samples else 0.0

    @property
    def avg_gpu(self):
        return float(np.mean(self.gpu_samples)) if self.gpu_samples else 0.0

    @property
    def peak_vram_gb(self):
        return float(np.max(self.vram_samples)) if self.vram_samples else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# NEUTRAL SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def suppress_neutral_batch(
    scores_list: list[dict[str, float]],
    strength: float = 1.0,
) -> list[dict[str, float]]:
    """Apply neutral suppression to a list of score dicts."""
    results = []
    for scores in scores_list:
        high_arousal_total = sum(scores.get(l, 0) for l in HIGH_AROUSAL_LABELS)
        neutral_score      = scores.get("neutral", 0)

        if high_arousal_total < 0.08 or neutral_score < 0.1:
            results.append(scores)
            continue

        reduction = neutral_score * strength * min(high_arousal_total * 3, 1.0)
        new       = dict(scores)
        new["neutral"] = max(0.0, neutral_score - reduction)

        non_neutral = {k: v for k, v in new.items() if k != "neutral" and v > 0}
        nn_total    = sum(non_neutral.values())
        if nn_total > 0:
            for k in non_neutral:
                new[k] += reduction * (non_neutral[k] / nn_total)

        # Renormalise
        total = sum(new.values())
        if total > 0:
            new = {k: v / total for k, v in new.items()}
        results.append(new)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE BLEND
# ─────────────────────────────────────────────────────────────────────────────

def blend_batch(
    go_list:       list[dict[str, float]],
    hartmann_list: list[dict[str, float]],
    weight:        float = 0.35,
) -> list[dict[str, float]]:
    blended_list = []
    for go, h in zip(go_list, hartmann_list):
        blended = dict(go)
        for h_label, h_score in h.items():
            children = HARTMANN_MAP.get(h_label, [])
            boost    = (h_score * weight) / max(len(children), 1)
            for child in children:
                if child in blended:
                    blended[child] = blended[child] + boost
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        blended_list.append(blended)
    return blended_list


# ─────────────────────────────────────────────────────────────────────────────
# VA CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def scores_to_va(scores: dict[str, float]) -> tuple[float, float]:
    """Weighted average VA position from a score dict."""
    v_total, a_total, w_total = 0.0, 0.0, 0.0
    for label, score in scores.items():
        if score <= 0:
            continue
        meta     = EMOTION_META.get(label)
        if meta is None:
            continue
        v_total += meta["v"] * score
        a_total += meta["a"] * score
        w_total += score
    if w_total == 0:
        return 0.0, 0.0
    return round(v_total / w_total, 4), round(a_total / w_total, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_book(
    book_id:             str,
    batch_size:          int   = 24,
    neutral_suppression: float = 1.0,
    ensemble_weight:     float = 0.35,
    min_score:           float = 0.05,
    device:              int   = 0,
) -> tuple[pd.DataFrame, dict]:

    meta = BOOKS[book_id]
    print(f"\n{'═'*60}")
    print(f"  {meta['title']}  —  {meta['author']}")
    print(f"{'═'*60}")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[1/5] Loading models…")
    t0 = time.perf_counter()

    go_pipe = pipeline(
        "text-classification",
        model    = "SamLowe/roberta-base-go_emotions",
        top_k    = None,
        device   = device,
        truncation=True,
        max_length=512,
    )
    h_pipe = pipeline(
        "text-classification",
        model    = "j-hartmann/emotion-english-distilroberta-base",
        top_k    = None,
        device   = device,
        truncation=True,
        max_length=512,
    )
    model_load_time = time.perf_counter() - t0
    print(f"   Models loaded in {model_load_time:.1f}s")

    # ── Fetch + clean book ────────────────────────────────────────────────────
    print("\n[2/5] Fetching book…")
    raw_text   = fetch_text(book_id)
    clean_text = strip_boilerplate(raw_text)

    # ── Tokenise sentences ────────────────────────────────────────────────────
    print("\n[3/5] Tokenising…")
    all_sentences = nltk.sent_tokenize(clean_text)
    # Filter very short / blank
    sentences  = [s.strip() for s in all_sentences if len(s.split()) >= 4]
    n_sentences = len(sentences)
    print(f"   {n_sentences:,} sentences")

    # Assign page numbers
    pages = assign_page_numbers(sentences, clean_text)
    total_pages = max(pages)
    print(f"   {total_pages} pages  ({LINES_PER_PAGE} lines/page)")

    # ── Start resource monitor ────────────────────────────────────────────────
    monitor = ResourceMonitor()
    monitor.start()

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Analysing  (batch={batch_size}) …")
    t_infer_start = time.perf_counter()

    go_results = []
    h_results  = []

    # Process in batches with tqdm progress bar
    batches   = range(0, n_sentences, batch_size)
    with tqdm(total=n_sentences, unit="sent", ncols=80,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} {rate_fmt}]"
              ) as pbar:
        for start in batches:
            batch = sentences[start: start + batch_size]

            # GoEmotions
            raw_go = go_pipe(batch)
            for item in raw_go:
                if isinstance(item, list):
                    go_results.append({e["label"]: e["score"] for e in item})
                else:
                    go_results.append({item["label"]: item["score"]})

            # Hartmann
            raw_h = h_pipe(batch)
            for item in raw_h:
                if isinstance(item, list):
                    h_results.append({e["label"]: e["score"] for e in item})
                else:
                    h_results.append({item["label"]: item["score"]})

            pbar.update(len(batch))

    t_infer = time.perf_counter() - t_infer_start
    monitor.stop()

    # ── Post-process ──────────────────────────────────────────────────────────
    print("\n[5/5] Post-processing…")

    # Ensemble blend
    blended = blend_batch(go_results, h_results, weight=ensemble_weight)

    # Neutral suppression
    suppressed = suppress_neutral_batch(blended, strength=neutral_suppression)

    # Build result rows
    rows = []
    for idx, (sent, page, scores) in enumerate(
            zip(sentences, pages, suppressed)):

        # Dominant emotion (above threshold)
        sorted_emotions = sorted(
            [(k, v) for k, v in scores.items() if v >= min_score],
            key=lambda x: x[1], reverse=True
        )
        dominant        = sorted_emotions[0][0] if sorted_emotions else "neutral"
        dominant_score  = sorted_emotions[0][1] if sorted_emotions else 0.0
        dominant_group  = EMOTION_META.get(dominant, {}).get("group", "Neutral / Flat")

        # VA
        va_v, va_a = scores_to_va(scores)

        row = {
            "sentence_idx":   idx,
            "page":           page,
            "sentence":       sent,
            "dominant":       dominant,
            "dominant_score": round(dominant_score, 4),
            "dominant_group": dominant_group,
            "va_v":           va_v,
            "va_a":           va_a,
        }
        # Individual emotion scores
        for label in ALL_LABELS:
            row[f"score_{label}"] = round(scores.get(label, 0.0), 5)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Stats ──────────────────────────────────────────────────────────────────
    total_wall = t_infer   # model load not included in throughput calc
    stats = {
        "book_id":              book_id,
        "title":                meta["title"],
        "author":               meta["author"],
        "timestamp":            datetime.now().isoformat(),
        "n_sentences":          n_sentences,
        "n_pages":              total_pages,
        "settings": {
            "neutral_suppression": neutral_suppression,
            "ensemble_weight":     ensemble_weight,
            "min_score":           min_score,
            "batch_size":          batch_size,
        },
        "performance": {
            "model_load_s":        round(model_load_time, 2),
            "inference_s":         round(t_infer, 2),
            "sentences_per_sec":   round(n_sentences / t_infer, 2),
            "pages_per_sec":       round(total_pages / t_infer, 4),
            "avg_cpu_pct":         round(monitor.avg_cpu, 1),
            "avg_gpu_vram_pct":    round(monitor.avg_gpu, 1),
            "peak_vram_gb":        round(monitor.peak_vram_gb, 2),
            "device":              f"cuda:{device}" if torch.cuda.is_available() else "cpu",
            "gpu_name":            torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else "N/A",
        },
    }
    return df, stats


# ─────────────────────────────────────────────────────────────────────────────
# SAVE DATA
# ─────────────────────────────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, stats: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", stats["title"].lower()).strip("_")

    # CSV — full sentence-level data
    csv_path = out_dir / f"{slug}_sentences.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Data saved → {csv_path}")

    # JSON — stats + per-page group summaries (lighter weight)
    # Build page-level group summary
    group_cols  = [g for g in TIMELINE_GROUPS]
    page_groups = (
        df.groupby("page")
          .apply(lambda g: {
              grp: float(
                  g[[f"score_{lbl}"
                     for lbl, m in EMOTION_META.items()
                     if m["group"] == grp]].values.mean()
              )
              for grp in group_cols
          })
          .reset_index()
    )
    stats["page_summaries"] = page_groups.set_index("page")[0].to_dict()

    json_path = out_dir / f"{slug}_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats saved → {json_path}")

    return slug, csv_path, json_path


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_book(df: pd.DataFrame, stats: dict, out_dir: Path, slug: str):
    """
    Three-panel plot:
      Top    — group emotion intensity across pages (rolling smoothed)
      Middle — weighted VA trajectory (score-weighted per sentence, not flat mean)
      Bottom — dominant emotion pie (neutral excluded) + group averages bar
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), facecolor="#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        height_ratios=[3, 2, 2],
        hspace=0.45,
        wspace=0.35,
    )

    panel_colour = "#16213e"
    text_colour  = "#eaeaea"
    muted_colour = "#8892a4"

    def rgba(r, g, b, a):
        """0-255 rgb + 0-1 alpha → matplotlib tuple."""
        return (r / 255, g / 255, b / 255, a)

    grid_colour      = rgba(255, 255, 255, 0.06)
    crosshair_colour = rgba(42,  74,  127, 0.80)

    # ── Page-level aggregates ─────────────────────────────────────────────────
    #
    # VA fix: weight each sentence's VA contribution by its dominant_score
    # so strongly-tagged sentences pull the page value more than weak ones.
    # This prevents cancellation from diluting the signal to near-zero.
    #
    page_data = {}
    for page, grp in df.groupby("page"):
        entry = {"page": page}

        # Group intensities (unchanged — flat mean is correct here)
        for g_name in TIMELINE_GROUPS:
            labels = [lbl for lbl, m in EMOTION_META.items()
                      if m["group"] == g_name]
            cols   = [f"score_{lbl}" for lbl in labels]
            entry[g_name] = float(grp[cols].values.mean())

        # Score-weighted VA — weight by dominant_score per sentence
        weights = grp["dominant_score"].values
        w_sum   = weights.sum()
        if w_sum > 0:
            entry["va_v"] = float(np.average(grp["va_v"].values, weights=weights))
            entry["va_a"] = float(np.average(grp["va_a"].values, weights=weights))
        else:
            entry["va_v"] = float(grp["va_v"].mean())
            entry["va_a"] = float(grp["va_a"].mean())

        page_data[page] = entry

    pages_sorted = sorted(page_data.keys())
    pg_df = pd.DataFrame([page_data[p] for p in pages_sorted])

    # Smoothing window ≈ 5% of pages, minimum 3
    window = max(3, len(pages_sorted) // 20)

    def smooth(series):
        return series.rolling(window, center=True, min_periods=1).mean()

    # ── Panel 1: Emotion group timeline ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(panel_colour)

    for g_name, hex_colour in TIMELINE_GROUPS.items():
        y = smooth(pg_df[g_name])
        ax1.plot(
            pages_sorted, y,
            color=hex_colour, linewidth=1.5,
            label=g_name.split(" / ")[0], alpha=0.9,
        )
        ax1.fill_between(pages_sorted, y, alpha=0.08, color=hex_colour)

    ax1.set_xlim(pages_sorted[0], pages_sorted[-1])
    ax1.set_xlabel("Page",                 color=muted_colour, fontsize=9)
    ax1.set_ylabel("Mean group intensity", color=muted_colour, fontsize=9)
    ax1.set_title(
        f"{stats['title']}  —  {stats['author']}\n"
        f"Emotion Timeline  ({stats['n_sentences']:,} sentences · "
        f"{stats['n_pages']} pages · "
        f"smoothing window={window})",
        color=text_colour, fontsize=11, pad=10,
    )
    ax1.tick_params(colors=muted_colour, labelsize=8)
    ax1.spines[:].set_color("#0f3460")
    ax1.legend(
        loc="upper right", fontsize=8,
        facecolor="#0f3460", edgecolor="#2a4a7f", labelcolor=text_colour,
    )
    ax1.grid(axis="y", color=grid_colour, linewidth=0.5)

    # ── Panel 2: VA trajectory ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor(panel_colour)

    v_smooth = smooth(pg_df["va_v"]).values
    a_smooth = smooth(pg_df["va_a"]).values

    ax2.plot(pages_sorted, v_smooth, color="#3498db", linewidth=1.6,
             label="Valence", alpha=0.9)
    ax2.plot(pages_sorted, a_smooth, color="#e74c3c", linewidth=1.6,
             label="Arousal", alpha=0.9)
    ax2.axhline(0, color=crosshair_colour, linewidth=0.8, linestyle="--")
    ax2.fill_between(pages_sorted, v_smooth, 0, alpha=0.08, color="#3498db")
    ax2.fill_between(pages_sorted, a_smooth, 0, alpha=0.08, color="#e74c3c")

    # Dynamic y-axis: show the actual range with a small margin
    all_va    = np.concatenate([v_smooth, a_smooth])
    va_margin = 0.08
    va_min    = min(all_va.min() - va_margin, -0.15)
    va_max    = max(all_va.max() + va_margin,  0.15)
    ax2.set_ylim(va_min, va_max)

    ax2.set_xlim(pages_sorted[0], pages_sorted[-1])
    ax2.set_xlabel("Page",              color=muted_colour, fontsize=9)
    ax2.set_ylabel("Weighted VA score", color=muted_colour, fontsize=9)
    ax2.set_title("Valence & Arousal Trajectory  (score-weighted per sentence)",
                  color=text_colour, fontsize=10, pad=8)
    ax2.tick_params(colors=muted_colour, labelsize=8)
    ax2.spines[:].set_color("#0f3460")
    ax2.legend(
        loc="upper right", fontsize=8,
        facecolor="#0f3460", edgecolor="#2a4a7f", labelcolor=text_colour,
    )
    ax2.grid(axis="y", color=grid_colour, linewidth=0.5)

    # Annotate actual range
    ax2.text(
        0.01, 0.04,
        f"V range [{v_smooth.min():.2f}, {v_smooth.max():.2f}]  "
        f"A range [{a_smooth.min():.2f}, {a_smooth.max():.2f}]",
        transform=ax2.transAxes,
        color=muted_colour, fontsize=7.5, va="bottom",
    )

    # ── Panel 3a: Dominant emotion pie — neutral excluded ─────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.set_facecolor(panel_colour)

    # Remove neutral and Neutral / Flat group from pie
    neutral_labels = {lbl for lbl, m in EMOTION_META.items()
                      if m["group"] == "Neutral / Flat"}
    dom_counts = df.loc[
        ~df["dominant"].isin(neutral_labels), "dominant"
    ].value_counts()

    # Hide slices < 1% of the non-neutral total
    threshold = dom_counts.sum() * 0.01
    dom_major = dom_counts[dom_counts >= threshold].copy()
    dom_other = dom_counts[dom_counts <  threshold].sum()
    if dom_other > 0:
        dom_major["other"] = dom_other

    pie_colours = [
        EMOTION_META.get(lbl, {}).get("colour", "#888888")
        for lbl in dom_major.index
    ]
    wedges, texts, autotexts = ax3.pie(
        dom_major.values,
        labels=dom_major.index,
        colors=pie_colours,
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=140,
        textprops={"color": text_colour, "fontsize": 7.5},
    )
    for at in autotexts:
        at.set_fontsize(7)
        at.set_color(text_colour)
    ax3.set_title("Dominant Emotion Distribution  (neutral excluded)",
                  color=text_colour, fontsize=10, pad=8)

    # ── Panel 3b: Group averages bar ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.set_facecolor(panel_colour)

    group_means = {}
    for g_name in TIMELINE_GROUPS:
        labels = [lbl for lbl, m in EMOTION_META.items()
                  if m["group"] == g_name]
        cols   = [f"score_{lbl}" for lbl in labels]
        group_means[g_name] = float(df[cols].values.mean())

    g_labels = [k.split(" / ")[0] for k in group_means]
    g_values = list(group_means.values())
    g_cols   = list(TIMELINE_GROUPS.values())

    bars = ax4.barh(g_labels, g_values, color=g_cols, alpha=0.85, height=0.5)
    ax4.set_xlabel("Mean score across book", color=muted_colour, fontsize=9)
    ax4.set_title("Group Averages (full book)",
                  color=text_colour, fontsize=10, pad=8)
    ax4.tick_params(colors=muted_colour, labelsize=9)
    ax4.spines[:].set_color("#0f3460")
    ax4.grid(axis="x", color=grid_colour, linewidth=0.5)

    for bar, val in zip(bars, g_values):
        ax4.text(
            val + 0.0005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left",
            color=text_colour, fontsize=8,
        )

    # ── Footer — same font as the rest of the figure (no monospace) ──────────
    perf = stats["performance"]
    s    = stats["settings"]
    footer = (
        f"Inference: {perf['inference_s']:.0f}s  ·  "
        f"{perf['sentences_per_sec']:.1f} sent/s  ·  "
        f"{perf['pages_per_sec']:.2f} pages/s  ·  "
        f"CPU avg {perf['avg_cpu_pct']:.0f}%  ·  "
        f"GPU VRAM avg {perf['avg_gpu_vram_pct']:.0f}%  ·  "
        f"Peak VRAM {perf['peak_vram_gb']:.2f} GB  ·  "
        f"{perf['gpu_name']}  ·  "
        f"ns={s['neutral_suppression']}  "
        f"ens={s['ensemble_weight']}  "
        f"batch={s['batch_size']}"
    )
    fig.text(
        0.5, 0.01, footer,
        ha="center", va="bottom",
        color=muted_colour, fontsize=8,
        # No fontfamily override — inherits the figure default (same as title)
    )

    plot_path = out_dir / f"{slug}_timeline.png"
    plt.savefig(
        plot_path, dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close()
    print(f"  Plot saved  → {plot_path}")
    return plot_path


# ─────────────────────────────────────────────────────────────────────────────
# PRINT STATS REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(stats: dict):
    """Cleanly aligned stats box using consistent column widths."""
    p = stats["performance"]
    s = stats["settings"]

    # Fixed column widths: label=26, value=14
    W_LABEL = 26
    W_VALUE = 14
    W_TOTAL = W_LABEL + W_VALUE   # 40 inner chars
    BORDER   = "─" * W_TOTAL

    def row(label, value, unit=""):
        v = f"{value}{unit}"
        return f"│  {label:<{W_LABEL}}{v:>{W_VALUE}}  │"

    def section(title):
        return (
            f"├{BORDER}┤\n"
            f"│  {title:<{W_TOTAL}}  │\n"
            f"├{BORDER}┤"
        )

    lines = [
        f"┌{BORDER}┐",
        f"│{'ANALYSIS COMPLETE':^{W_TOTAL + 2}}│",
        f"├{BORDER}┤",
        row("Book",   stats["title"]),
        row("Author", stats["author"]),
        section("THROUGHPUT"),
        row("Sentences processed",   f"{stats['n_sentences']:,}"),
        row("Pages processed",       f"{stats['n_pages']:,}"),
        row("Inference time",        f"{p['inference_s']:.1f}",  "s"),
        row("Model load time",       f"{p['model_load_s']:.1f}", "s"),
        row("Sentences / second",    f"{p['sentences_per_sec']:.2f}"),
        row("Pages / second",        f"{p['pages_per_sec']:.4f}"),
        section("RESOURCE USAGE"),
        row("Device",               p["device"]),
        row("GPU",                  p["gpu_name"]),
        row("CPU avg utilisation",  f"{p['avg_cpu_pct']:.1f}",  "%"),
        row("GPU VRAM avg",         f"{p['avg_gpu_vram_pct']:.1f}", "%"),
        row("Peak VRAM",            f"{p['peak_vram_gb']:.2f}", " GB"),
        section("SETTINGS"),
        row("Neutral suppression",  f"{s['neutral_suppression']:.1f}"),
        row("Ensemble weight",      f"{s['ensemble_weight']:.2f}"),
        row("Min score",            f"{s['min_score']:.2f}"),
        row("Batch size",           f"{s['batch_size']}"),
        f"└{BORDER}┘",
    ]
    print("\n" + "\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full-book emotion analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--book", default="345",
        help="Gutenberg book ID  (default: 345 = Dracula)",
    )
    parser.add_argument(
        "--batch", type=int, default=24,
        help="Inference batch size  (default: 24, raise if VRAM allows)",
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device index  (default: 0, use -1 for CPU)",
    )
    parser.add_argument(
        "--out", default="output",
        help="Output directory  (default: ./output)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available books and exit",
    )
    # Override defaults
    parser.add_argument("--ns",  type=float, default=1.0,
                        help="Neutral suppression strength  (default: 1.0)")
    parser.add_argument("--ew",  type=float, default=0.35,
                        help="Ensemble (Hartmann) weight  (default: 0.35)")
    parser.add_argument("--min", type=float, default=0.05,
                        help="Minimum score threshold  (default: 0.05)")

    args = parser.parse_args()

    if args.list:
        print("\nAvailable books:")
        for bid, meta in BOOKS.items():
            print(f"  {bid:>6}  {meta['title']}  —  {meta['author']}")
        sys.exit(0)

    if args.book not in BOOKS:
        print(f"Error: book ID '{args.book}' not in catalogue. Use --list.")
        sys.exit(1)

    out_dir = Path(args.out)

    # ── Run ────────────────────────────────────────────────────────────────────
    df, stats = analyse_book(
        book_id             = args.book,
        batch_size          = args.batch,
        neutral_suppression = args.ns,
        ensemble_weight     = args.ew,
        min_score           = args.min,
        device              = args.device,
    )

    # ── Save ───────────────────────────────────────────────────────────────────
    print("\n[Saving results…]")
    slug, csv_path, json_path = save_results(df, stats, out_dir)

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("[Plotting…]")
    plot_path = plot_book(df, stats, out_dir, slug)

    # ── Stats report ───────────────────────────────────────────────────────────
    print_stats(stats)

    print(f"\n  All done.  Output in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()