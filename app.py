import os
import re
import json
import requests
import nltk
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformers import pipeline
from bs4 import BeautifulSoup

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Model loading ─────────────────────────────────────────────────────────────
print("Loading GoEmotions (RoBERTa)…")
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,
    device=0,
    truncation=True,
    max_length=512,
)

print("Loading j-hartmann ensemble model…")
hartmann_classifier = pipeline(
    task="text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=0,
    truncation=True,
    max_length=512,
)
print("Both models loaded.")

# ── Hartmann → GoEmotions label mapping ───────────────────────────────────────
# j-hartmann outputs: anger, disgust, fear, joy, neutral, sadness, surprise
# We map these onto our GoEmotions labels for blending
HARTMANN_MAP = {
    "anger":    ["anger", "annoyance"],
    "disgust":  ["disgust", "disapproval"],
    "fear":     ["fear", "nervousness"],
    "joy":      ["joy", "excitement", "amusement"],
    "neutral":  ["neutral"],
    "sadness":  ["sadness", "grief", "remorse"],
    "surprise": ["surprise", "realization"],
}

# ── Emotion metadata ───────────────────────────────────────────────────────────
EMOTION_GROUPS = {
    "fear":           {"colour": "#c0392b", "group": "Tension / Danger"},
    "nervousness":    {"colour": "#e74c3c", "group": "Tension / Danger"},
    "anger":          {"colour": "#e67e22", "group": "Tension / Danger"},
    "annoyance":      {"colour": "#f39c12", "group": "Tension / Danger"},
    "disgust":        {"colour": "#8e44ad", "group": "Drama / Darkness"},
    "grief":          {"colour": "#6c3483", "group": "Drama / Darkness"},
    "sadness":        {"colour": "#5b2c6f", "group": "Drama / Darkness"},
    "remorse":        {"colour": "#76448a", "group": "Drama / Darkness"},
    "joy":            {"colour": "#27ae60", "group": "Positive / Uplifting"},
    "love":           {"colour": "#e91e63", "group": "Positive / Uplifting"},
    "admiration":     {"colour": "#2ecc71", "group": "Positive / Uplifting"},
    "excitement":     {"colour": "#f1c40f", "group": "Positive / Uplifting"},
    "amusement":      {"colour": "#1abc9c", "group": "Positive / Uplifting"},
    "gratitude":      {"colour": "#16a085", "group": "Positive / Uplifting"},
    "pride":          {"colour": "#148f77", "group": "Positive / Uplifting"},
    "optimism":       {"colour": "#45b39d", "group": "Positive / Uplifting"},
    "relief":         {"colour": "#52be80", "group": "Positive / Uplifting"},
    "surprise":       {"colour": "#3498db", "group": "Contemplative"},
    "curiosity":      {"colour": "#2980b9", "group": "Contemplative"},
    "realization":    {"colour": "#1f618d", "group": "Contemplative"},
    "confusion":      {"colour": "#5d6d7e", "group": "Contemplative"},
    "approval":       {"colour": "#85c1e9", "group": "Contemplative"},
    "desire":         {"colour": "#a9cce3", "group": "Contemplative"},
    "neutral":        {"colour": "#95a5a6", "group": "Neutral / Flat"},
    "caring":         {"colour": "#abebc6", "group": "Neutral / Flat"},
    "disapproval":    {"colour": "#aab7b8", "group": "Neutral / Flat"},
    "embarrassment":  {"colour": "#d7bde2", "group": "Neutral / Flat"},
    "disappointment": {"colour": "#a2d9ce", "group": "Neutral / Flat"},
}

# High-arousal labels — used to decide whether neutral suppression fires
HIGH_AROUSAL_LABELS = {
    "fear", "anger", "nervousness", "excitement",
    "disgust", "grief", "sadness", "surprise",
}

TOP_N_EMOTIONS      = 3
BASE_THRESHOLD      = 0.05


# ── Gutenberg helpers ─────────────────────────────────────────────────────────
SAMPLE_BOOKS = [
    {"id": "1342", "title": "Pride and Prejudice",          "author": "Jane Austen",
     "url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"},
    {"id": "2701", "title": "Moby Dick",                    "author": "Herman Melville",
     "url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"},
    {"id": "174",  "title": "The Picture of Dorian Gray",   "author": "Oscar Wilde",
     "url": "https://www.gutenberg.org/cache/epub/174/pg174.txt"},
    {"id": "1661", "title": "The Adventures of Sherlock Holmes", "author": "Arthur Conan Doyle",
     "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"},
    {"id": "84",   "title": "Frankenstein",                 "author": "Mary Shelley",
     "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt"},
    {"id": "345",  "title": "Dracula",                      "author": "Bram Stoker",
     "url": "https://www.gutenberg.org/cache/epub/345/pg345.txt"},
    {"id": "1400", "title": "Great Expectations",           "author": "Charles Dickens",
     "url": "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"},
]


def strip_gutenberg_boilerplate(text: str) -> str:
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
    ]
    start_idx = 0
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            start_idx = text.index("\n", idx) + 1
            break
    end_idx = len(text)
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            end_idx = idx
            break
    return text[start_idx:end_idx].strip()


def extract_page(text: str, page: int, lines_per_page: int = 40) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    start = (page - 1) * lines_per_page
    end   =  page      * lines_per_page
    return "\n".join(lines[start:end])


def count_pages(text: str, lines_per_page: int = 40) -> int:
    lines = [l for l in text.splitlines() if l.strip()]
    return max(1, (len(lines) + lines_per_page - 1) // lines_per_page)


_book_cache: dict[str, str] = {}


def fetch_book_text(url: str) -> str:
    if url in _book_cache:
        return _book_cache[url]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1")
    cleaned = strip_gutenberg_boilerplate(text)
    _book_cache[url] = cleaned
    return cleaned


# ── Core classification helpers ───────────────────────────────────────────────

def classify_single(text: str) -> dict[str, float]:
    """Run GoEmotions on a single text chunk. Returns {label: score}."""
    raw = emotion_classifier(text[:512])
    if isinstance(raw[0], list):
        raw = raw[0]
    return {e["label"]: e["score"] for e in raw}


def classify_hartmann(text: str) -> dict[str, float]:
    """Run Hartmann model and return {label: score}."""
    raw = hartmann_classifier(text[:512])
    if isinstance(raw[0], list):
        raw = raw[0]
    return {e["label"]: e["score"] for e in raw}


def blend_with_hartmann(
    go_scores: dict[str, float],
    text: str,
    blend_weight: float = 0.35,
) -> dict[str, float]:
    """
    Blend GoEmotions scores with Hartmann scores.
    Hartmann's cleaner 6-class output acts as a corrective signal.
    blend_weight: how much Hartmann contributes (0 = GoEmotions only).
    """
    h_scores = classify_hartmann(text)

    # Build a correction map: for each GoEmotions label, find the
    # Hartmann parent score and add a proportion of it
    blended = dict(go_scores)
    for h_label, h_score in h_scores.items():
        children = HARTMANN_MAP.get(h_label, [])
        boost    = (h_score * blend_weight) / max(len(children), 1)
        for child in children:
            if child in blended:
                blended[child] = blended[child] + boost

    # Renormalise so scores sum to ~1
    total = sum(blended.values())
    if total > 0:
        blended = {k: v / total for k, v in blended.items()}
    return blended


def suppress_neutral(
    scores: dict[str, float],
    suppression_strength: float = 0.5,
) -> dict[str, float]:
    """
    If high-arousal emotions have meaningful combined score,
    reduce neutral proportionally and redistribute to non-neutral labels.

    suppression_strength: 0 = no suppression, 1 = full suppression of neutral.
    """
    high_arousal_total = sum(
        scores.get(lbl, 0) for lbl in HIGH_AROUSAL_LABELS
    )

    # Only fire if there's meaningful high-arousal signal
    if high_arousal_total < 0.08:
        return scores

    neutral_score = scores.get("neutral", 0)
    if neutral_score < 0.1:
        return scores

    # How much to take from neutral
    reduction = neutral_score * suppression_strength * min(high_arousal_total * 3, 1.0)
    new_scores = dict(scores)
    new_scores["neutral"] = max(0, neutral_score - reduction)

    # Redistribute proportionally to non-neutral labels
    non_neutral = {
        k: v for k, v in new_scores.items()
        if k != "neutral" and v > 0
    }
    non_neutral_total = sum(non_neutral.values())
    if non_neutral_total > 0:
        for k in non_neutral:
            new_scores[k] += reduction * (non_neutral[k] / non_neutral_total)

    return new_scores


def apply_context_window(
    segments: list[str],
    idx: int,
    window: int,
) -> str:
    """
    Return a string of [window] sentences before and after idx,
    with the target sentence marked so the model sees narrative context.
    """
    start = max(0, idx - window)
    end   = min(len(segments), idx + window + 1)

    parts = []
    for i in range(start, end):
        if i == idx:
            parts.append(f"[TARGET] {segments[i]} [/TARGET]")
        else:
            parts.append(segments[i])

    return " ".join(parts)


# ── Main analysis function ────────────────────────────────────────────────────

def analyse_text(
    text: str,
    granularity: str        = "sentence",
    context_window: int     = 0,       # 0 = disabled
    use_ensemble: bool      = False,
    ensemble_weight: float  = 0.35,
    neutral_suppression: float = 0.0,  # 0 = off, 0.5 = moderate, 1.0 = aggressive
    threshold: float        = BASE_THRESHOLD,
) -> list[dict]:

    if granularity == "sentence":
        segments = nltk.sent_tokenize(text)
    else:
        segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    results = []
    for idx, seg in enumerate(segments):
        if len(seg.split()) < 3:
            continue

        # Build the text to actually classify (with optional context)
        if context_window > 0 and granularity == "sentence":
            classify_text = apply_context_window(segments, idx, context_window)
        else:
            classify_text = seg

        # GoEmotions base scores
        go_scores = classify_single(classify_text[:512])

        # Optional ensemble blend
        if use_ensemble:
            # Blend on the raw segment (not context-padded) for Hartmann
            # as it doesn't benefit from the [TARGET] markers
            go_scores = blend_with_hartmann(go_scores, seg, ensemble_weight)

        # Optional neutral suppression
        if neutral_suppression > 0:
            go_scores = suppress_neutral(go_scores, neutral_suppression)

        # Filter, sort, top-N
        filtered = [
            {"label": k, "score": v}
            for k, v in go_scores.items()
            if v >= threshold
        ]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        top = filtered[:TOP_N_EMOTIONS]

        emotions = []
        for e in top:
            label = e["label"]
            score = round(e["score"], 4)
            meta  = EMOTION_GROUPS.get(label, {"colour": "#bdc3c7", "group": "Neutral / Flat"})
            emotions.append({
                "label":  label,
                "score":  score,
                "colour": meta["colour"],
                "group":  meta["group"],
            })

        results.append({
            "text":     seg,          # always display the original segment
            "emotions": emotions,
            "dominant": emotions[0] if emotions else None,
        })

    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", books=SAMPLE_BOOKS)


@app.route("/api/books")
def api_books():
    return jsonify(SAMPLE_BOOKS)


@app.route("/api/fetch-book")
def api_fetch_book():
    book_id = request.args.get("id")
    book    = next((b for b in SAMPLE_BOOKS if b["id"] == book_id), None)
    if not book:
        return jsonify({"error": "Book not found"}), 404
    try:
        text = fetch_book_text(book["url"])
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    page        = int(request.args.get("page", 1))
    total_pages = count_pages(text)
    page_text   = extract_page(text, page)

    return jsonify({
        "title":       book["title"],
        "author":      book["author"],
        "page":        page,
        "total_pages": total_pages,
        "text":        page_text,
    })


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json()

    text                = data.get("text", "")
    granularity         = data.get("granularity", "sentence")
    context_window      = int(data.get("context_window", 0))
    use_ensemble        = bool(data.get("use_ensemble", False))
    ensemble_weight     = float(data.get("ensemble_weight", 0.35))
    neutral_suppression = float(data.get("neutral_suppression", 0.0))
    threshold           = float(data.get("threshold", BASE_THRESHOLD))

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        segments = analyse_text(
            text,
            granularity         = granularity,
            context_window      = context_window,
            use_ensemble        = use_ensemble,
            ensemble_weight     = ensemble_weight,
            neutral_suppression = neutral_suppression,
            threshold           = threshold,
        )
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(ex)}), 500

    return jsonify({"segments": segments})


if __name__ == "__main__":
    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab")
    app.run(debug=False, port=5000)