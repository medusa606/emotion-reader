import os
import re
import json
import requests
import nltk
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from transformers import pipeline
from bs4 import BeautifulSoup

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Emotion model ─────────────────────────────────────────────────────────────
print("Loading emotion model... (first run will download ~500MB)")
emotion_classifier = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None,           # return all scores
    device=0,             # GPU - change to -1 for CPU
    truncation=True,
    max_length=512
)
print("Model loaded.")

# ── Emotion → colour mapping ──────────────────────────────────────────────────
# Grouped by narrative "feel" for the reader UI
EMOTION_GROUPS = {
    # Tension / Danger
    "fear":        {"colour": "#c0392b", "group": "tension"},
    "nervousness": {"colour": "#e74c3c", "group": "tension"},
    "anger":       {"colour": "#e67e22", "group": "tension"},
    "annoyance":   {"colour": "#f39c12", "group": "tension"},

    # Drama / Intensity
    "disgust":     {"colour": "#8e44ad", "group": "drama"},
    "grief":       {"colour": "#6c3483", "group": "drama"},
    "sadness":     {"colour": "#5b2c6f", "group": "drama"},
    "remorse":     {"colour": "#76448a", "group": "drama"},

    # Positive / Uplifting
    "joy":         {"colour": "#27ae60", "group": "positive"},
    "love":        {"colour": "#e91e63", "group": "positive"},
    "admiration":  {"colour": "#2ecc71", "group": "positive"},
    "excitement":  {"colour": "#f1c40f", "group": "positive"},
    "amusement":   {"colour": "#1abc9c", "group": "positive"},
    "gratitude":   {"colour": "#16a085", "group": "positive"},
    "pride":       {"colour": "#148f77", "group": "positive"},
    "optimism":    {"colour": "#45b39d", "group": "positive"},
    "relief":      {"colour": "#52be80", "group": "positive"},

    # Contemplative / Neutral
    "surprise":    {"colour": "#3498db", "group": "contemplative"},
    "curiosity":   {"colour": "#2980b9", "group": "contemplative"},
    "realization": {"colour": "#1f618d", "group": "contemplative"},
    "confusion":   {"colour": "#5d6d7e", "group": "contemplative"},
    "approval":    {"colour": "#85c1e9", "group": "contemplative"},
    "desire":      {"colour": "#a9cce3", "group": "contemplative"},

    # Flat / Neutral
    "neutral":     {"colour": "#95a5a6", "group": "neutral"},
    "caring":      {"colour": "#abebc6", "group": "neutral"},
    "disapproval": {"colour": "#aab7b8", "group": "neutral"},
    "embarrassment":{"colour": "#d7bde2","group": "neutral"},
    "disappointment":{"colour": "#a2d9ce","group": "neutral"},
}

TOP_N_EMOTIONS = 3          # emotions to return per segment
THRESHOLD     = 0.05        # minimum score to surface an emotion


# ── Gutenberg helpers ─────────────────────────────────────────────────────────
GUTENBERG_MIRROR = "https://www.gutenberg.org"

# Curated list: (title, author, text_url)
SAMPLE_BOOKS = [
    {
        "id":     "1342",
        "title":  "Pride and Prejudice",
        "author": "Jane Austen",
        "url":    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    },
    {
        "id":     "2701",
        "title":  "Moby Dick",
        "author": "Herman Melville",
        "url":    "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    },
    {
        "id":     "174",
        "title":  "The Picture of Dorian Gray",
        "author": "Oscar Wilde",
        "url":    "https://www.gutenberg.org/cache/epub/174/pg174.txt",
    },
    {
        "id":     "1661",
        "title":  "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "url":    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
    },
    {
        "id":     "84",
        "title":  "Frankenstein",
        "author": "Mary Shelley",
        "url":    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    },
    {
        "id":     "345",
        "title":  "Dracula",
        "author": "Bram Stoker",
        "url":    "https://www.gutenberg.org/cache/epub/345/pg345.txt",
    },
    {
        "id":     "1400",
        "title":  "Great Expectations",
        "author": "Charles Dickens",
        "url":    "https://www.gutenberg.org/cache/epub/1400/pg1400.txt",
    },
]


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Gutenberg header/footer boilerplate."""
    # Find start marker
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
            # skip to end of that line
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
    """
    Split cleaned text into pseudo-pages of `lines_per_page` non-empty lines.
    Returns the text of the requested page.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    start = (page - 1) * lines_per_page
    end   =  page      * lines_per_page
    page_lines = lines[start:end]
    if not page_lines:
        return ""
    return "\n".join(page_lines)


def count_pages(text: str, lines_per_page: int = 40) -> int:
    lines = [l for l in text.splitlines() if l.strip()]
    return max(1, (len(lines) + lines_per_page - 1) // lines_per_page)


# Simple in-memory cache so we don't hammer Gutenberg
_book_cache: dict[str, str] = {}


def fetch_book_text(url: str) -> str:
    if url in _book_cache:
        return _book_cache[url]
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    # Gutenberg .txt files are latin-1 or utf-8
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("latin-1")
    cleaned = strip_gutenberg_boilerplate(text)
    _book_cache[url] = cleaned
    return cleaned


# ── Emotion analysis ──────────────────────────────────────────────────────────
def analyse_text(text: str, granularity: str = "sentence"):
    """
    Split text into segments by sentence or paragraph,
    run emotion classifier, return annotated segments.
    """
    if granularity == "sentence":
        segments = nltk.sent_tokenize(text)
    else:
        # paragraph: split on blank lines
        segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    results = []
    for seg in segments:
        if len(seg.split()) < 3:          # skip very short fragments
            continue

        raw = emotion_classifier(seg[:512])   # list of {label, score}
        # raw comes back as [[{label, score}, ...]] from top_k=None
        if isinstance(raw[0], list):
            raw = raw[0]

        # Filter and sort
        filtered = [e for e in raw if e["score"] >= THRESHOLD]
        filtered.sort(key=lambda x: x["score"], reverse=True)
        top = filtered[:TOP_N_EMOTIONS]

        emotions = []
        for e in top:
            label = e["label"]
            score = round(e["score"], 4)
            meta  = EMOTION_GROUPS.get(label, {"colour": "#bdc3c7", "group": "neutral"})
            emotions.append({
                "label":  label,
                "score":  score,
                "colour": meta["colour"],
                "group":  meta["group"],
            })

        results.append({
            "text":    seg,
            "emotions": emotions,
            # dominant emotion for background tinting
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

    page       = int(request.args.get("page", 1))
    total_pages = count_pages(text)
    page_text  = extract_page(text, page)

    return jsonify({
        "title":       book["title"],
        "author":      book["author"],
        "page":        page,
        "total_pages": total_pages,
        "text":        page_text,
    })


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data        = request.get_json()
    text        = data.get("text", "")
    granularity = data.get("granularity", "sentence")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        segments = analyse_text(text, granularity)
    except Exception as ex:
        return jsonify({"error": str(ex)}), 500

    return jsonify({"segments": segments})


if __name__ == "__main__":
    app.run(debug=False, port=5000)