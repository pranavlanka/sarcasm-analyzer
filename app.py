"""
Sarcasm & Sentiment Analyzer
Covers: Semantics, Pragmatics, Discourse analysis
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    text: str
    surface_sentiment: str        # Naive sentiment (lexicon only)
    surface_score: float          # -1.0 to +1.0
    sarcasm_detected: bool
    sarcasm_confidence: float     # 0.0 to 1.0
    sarcasm_signals: list[str]
    true_sentiment: str           # Final sentiment after pragmatics
    true_score: float
    discourse_context: Optional[str]
    explanation: str
    layer_breakdown: dict


# ─────────────────────────────────────────────
# Lexicons
# ─────────────────────────────────────────────

POSITIVE_WORDS = {
    "great": 0.8, "amazing": 0.9, "wonderful": 0.85, "fantastic": 0.9,
    "love": 0.8, "excellent": 0.85, "brilliant": 0.8, "perfect": 0.9,
    "happy": 0.75, "awesome": 0.85, "good": 0.6, "nice": 0.6,
    "beautiful": 0.8, "outstanding": 0.85, "superb": 0.85, "lovely": 0.75,
    "enjoy": 0.7, "liked": 0.65, "best": 0.8, "fun": 0.7,
    "glad": 0.7, "pleased": 0.65, "terrific": 0.8, "cool": 0.6,
    "wow": 0.7, "yay": 0.75, "excited": 0.75, "delighted": 0.8,
    "thrilled": 0.8, "fabulous": 0.85, "impressive": 0.75, "helpful": 0.7,
}

NEGATIVE_WORDS = {
    "bad": -0.6, "terrible": -0.85, "horrible": -0.9, "awful": -0.85,
    "hate": -0.8, "worst": -0.85, "ugly": -0.7, "disgusting": -0.85,
    "stupid": -0.7, "boring": -0.65, "useless": -0.7, "fail": -0.75,
    "failed": -0.75, "broken": -0.7, "annoying": -0.65, "pain": -0.6,
    "miserable": -0.8, "dreadful": -0.85, "pathetic": -0.8, "disappointing": -0.7,
    "disappointed": -0.7, "disaster": -0.85, "nightmare": -0.8, "ugh": -0.6,
    "frustrated": -0.7, "upset": -0.65, "angry": -0.75, "waste": -0.7,
    "hard": -0.4, "difficult": -0.4, "stress": -0.65, "tired": -0.5,
    "exhausted": -0.65, "sad": -0.7, "depressed": -0.8, "unfortunate": -0.6,
}

INTENSIFIERS = {
    "very": 1.3, "really": 1.25, "so": 1.2, "extremely": 1.5,
    "absolutely": 1.4, "totally": 1.3, "completely": 1.35, "utterly": 1.4,
    "incredibly": 1.4, "insanely": 1.35, "super": 1.25, "quite": 1.1,
    "rather": 1.1, "pretty": 1.1, "just": 1.0, "literally": 1.15,
}

NEGATORS = {"not", "no", "never", "nobody", "nothing", "neither", "nor",
            "without", "hardly", "barely", "scarcely", "don't", "doesn't",
            "didn't", "won't", "wouldn't", "can't", "couldn't", "isn't",
            "aren't", "wasn't", "weren't", "shouldn't"}

# Sarcasm-triggering contexts
SARCASM_CONTEXTS = {
    "negative_events": [
        "exam", "exams", "test", "tests", "homework", "assignment", "deadline",
        "monday", "traffic", "rain", "cold", "sick", "broke", "failed", "late",
        "meeting", "presentation", "dentist", "taxes", "bill", "bills", "lost",
        "crash", "error", "bug", "rejection", "fired", "delay", "cancelled",
        "broken", "spilled", "burned", "forgot", "missed"
    ],
    "complaint_starters": [
        "another", "yet another", "one more", "again", "as if", "because",
        "just what", "just what i needed", "exactly what", "oh look",
        "oh great", "oh wonderful", "oh fantastic", "because obviously",
        "because clearly", "totally", "definitely", "absolutely"
    ],
    "forced_positivity": [
        "just love", "totally love", "absolutely love", "really love",
        "so happy", "really happy", "can't wait", "totally can't wait",
        "i'm fine", "everything's fine", "all good", "no problem",
        "sure thing", "oh sure", "yeah right", "right sure"
    ]
}

SARCASM_EMOJIS = ["😒", "🙄", "😑", "😤", "🤦", "😅", "💀", "🫠", "😩", "😫", "🙃"]

DISCOURSE_NEGATIVE_TRIGGERS = [
    "hate", "dislike", "annoyed", "frustrated", "tired of", "sick of",
    "so stressed", "having a bad", "terrible day", "worst day",
    "can't stand", "failed", "failed my", "lost my", "broke my",
    "forgot", "running late", "missed the", "got rejected"
]


# ─────────────────────────────────────────────
# Layer 1: Semantic Analysis (Lexicon-based)
# ─────────────────────────────────────────────

def analyze_semantics(text: str) -> tuple[float, dict]:
    """Compute surface-level sentiment score using lexicon + intensifiers + negation."""
    tokens = re.findall(r"[\w']+|[^\w\s]", text.lower())
    score = 0.0
    matched = []
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Check negation window
        negated = any(tokens[max(0, i-3):i][j] in NEGATORS
                      for j in range(len(tokens[max(0, i-3):i])))
        
        # Intensifier in previous token
        intensity = 1.0
        if i > 0 and tokens[i-1] in INTENSIFIERS:
            intensity = INTENSIFIERS[tokens[i-1]]
        
        if token in POSITIVE_WORDS:
            val = POSITIVE_WORDS[token] * intensity
            val = -val * 0.7 if negated else val
            score += val
            matched.append(f"'{token}' → {val:+.2f}")
        elif token in NEGATIVE_WORDS:
            val = NEGATIVE_WORDS[token] * intensity
            val = -val * 0.7 if negated else val
            score += val
            matched.append(f"'{token}' → {val:+.2f}")
        
        i += 1
    
    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))
    
    label = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"
    return score, {"label": label, "score": score, "matched_words": matched}


# ─────────────────────────────────────────────
# Layer 2: Pragmatic Analysis (Sarcasm Detection)
# ─────────────────────────────────────────────

def detect_sarcasm(text: str, surface_score: float) -> tuple[bool, float, list[str]]:
    """Detect sarcasm via pragmatic signals."""
    signals = []
    confidence = 0.0
    lower = text.lower()
    
    # Signal 1: Positive words + negative event context
    has_positive = surface_score > 0.2
    neg_event_hits = [w for w in SARCASM_CONTEXTS["negative_events"] if w in lower]
    if has_positive and neg_event_hits:
        signals.append(f"Positive words paired with negative event(s): {neg_event_hits}")
        confidence += 0.35
    
    # Signal 2: Sarcasm emojis
    emoji_hits = [e for e in SARCASM_EMOJIS if e in text]
    if emoji_hits:
        signals.append(f"Sarcasm-marker emoji(s) detected: {emoji_hits}")
        confidence += 0.30
    
    # Signal 3: Complaint starters (pragmatic framing)
    for phrase in SARCASM_CONTEXTS["complaint_starters"]:
        if lower.startswith(phrase) or f" {phrase}" in lower:
            signals.append(f"Complaint/irony starter phrase: '{phrase}'")
            confidence += 0.20
            break
    
    # Signal 4: Forced positivity patterns
    for phrase in SARCASM_CONTEXTS["forced_positivity"]:
        if phrase in lower:
            signals.append(f"Forced positivity pattern: '{phrase}'")
            confidence += 0.20
            break
    
    # Signal 5: Excessive punctuation / caps (written sarcasm tell)
    if re.search(r'[!?]{2,}', text):
        signals.append("Excessive punctuation (!!/??) — written sarcasm marker")
        confidence += 0.10
    
    if sum(1 for c in text if c.isupper()) > len(text) * 0.4 and len(text) > 5:
        signals.append("Heavy CAPS usage — sarcastic exaggeration")
        confidence += 0.15
    
    # Signal 6: "oh" + positive (classic sarcasm opener)
    if re.search(r'\boh\b.{0,10}(great|wonderful|amazing|perfect|fantastic)', lower):
        signals.append("'Oh [positive word]' construction — classic sarcasm pattern")
        confidence += 0.30
    
    # Signal 7: "yeah right" / "sure" dismissals
    if re.search(r'\b(yeah right|sure|oh sure|totally|as if)\b', lower):
        signals.append("Dismissive affirmative ('yeah right', 'sure', 'totally') detected")
        confidence += 0.20
    
    confidence = min(1.0, confidence)
    detected = confidence >= 0.35
    
    return detected, confidence, signals


# ─────────────────────────────────────────────
# Layer 3: Discourse Analysis (Context)
# ─────────────────────────────────────────────

def analyze_discourse(text: str, previous: Optional[str]) -> tuple[float, str]:
    """
    Use prior conversational turn to adjust sentiment.
    Returns a discourse modifier and explanation.
    """
    if not previous:
        return 0.0, "No prior context provided."
    
    lower_prev = previous.lower()
    negative_triggers = [t for t in DISCOURSE_NEGATIVE_TRIGGERS if t in lower_prev]
    
    if negative_triggers:
        return -0.25, (
            f"Prior message contains negative context "
            f"({', '.join(negative_triggers[:2])}), "
            f"suggesting current message is likely ironic/sarcastic."
        )
    
    # Check if prior context is also positive (genuine positivity chain)
    prev_score, _ = analyze_semantics(previous)
    if prev_score > 0.3:
        return 0.1, "Prior context is genuinely positive — current sentiment likely authentic."
    
    return 0.0, "Prior context is neutral; no strong discourse signal."


# ─────────────────────────────────────────────
# Main Analyzer
# ─────────────────────────────────────────────

def analyze(text: str, previous_context: Optional[str] = None) -> AnalysisResult:
    """Full 3-layer NLP analysis pipeline."""
    
    # Layer 1 – Semantics
    surface_score, semantic_detail = analyze_semantics(text)
    surface_label = semantic_detail["label"]
    
    # Layer 2 – Pragmatics (Sarcasm)
    sarcasm_detected, sarcasm_confidence, sarcasm_signals = detect_sarcasm(text, surface_score)
    
    # Layer 3 – Discourse
    discourse_mod, discourse_note = analyze_discourse(text, previous_context)
    
    # Combine into true sentiment
    true_score = surface_score
    
    if sarcasm_detected:
        # Flip and dampen the score
        true_score = -(surface_score * 0.8) + discourse_mod
    else:
        true_score = surface_score + discourse_mod
    
    # Discourse alone can push neutral → negative (context-induced irony)
    if not sarcasm_detected and discourse_mod < -0.1 and surface_score > 0.1:
        true_score = surface_score * 0.3 + discourse_mod
    
    true_score = max(-1.0, min(1.0, true_score))
    true_label = "Positive" if true_score > 0.15 else "Negative" if true_score < -0.15 else "Neutral"
    
    # Build explanation
    if sarcasm_detected:
        explanation = (
            f"Surface words appear {surface_label.lower()}, but pragmatic analysis "
            f"detects sarcasm (confidence: {sarcasm_confidence:.0%}). "
            f"True intent is likely {true_label.lower()}."
        )
    else:
        explanation = (
            f"No strong sarcasm detected. Surface sentiment ({surface_label}) "
            f"reflects true meaning. {discourse_note}"
        )
    
    return AnalysisResult(
        text=text,
        surface_sentiment=surface_label,
        surface_score=round(surface_score, 3),
        sarcasm_detected=sarcasm_detected,
        sarcasm_confidence=round(sarcasm_confidence, 3),
        sarcasm_signals=sarcasm_signals,
        true_sentiment=true_label,
        true_score=round(true_score, 3),
        discourse_context=previous_context,
        explanation=explanation,
        layer_breakdown={
            "semantics": semantic_detail,
            "pragmatics": {
                "sarcasm_detected": sarcasm_detected,
                "confidence": sarcasm_confidence,
                "signals": sarcasm_signals
            },
            "discourse": {
                "modifier": discourse_mod,
                "note": discourse_note
            }
        }
    )


# ─────────────────────────────────────────────
# CLI Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    examples = [
        ("Great, another exam 😒", None),
        ("Oh wonderful, the server crashed again!!", None),
        ("I love this product! It works perfectly.", None),
        ("Yeah right, totally loving this Monday traffic 🙄", None),
        ("The movie was actually really good.", None),
        ("So happy my flight got cancelled. AMAZING.", None),
        ("I had a terrible day, everything went wrong.", "I just failed my exam and lost my keys"),
        ("Oh great, more homework!", "I hate this semester so much"),
        ("This is fine.", None),
    ]
    
    print("=" * 65)
    print("   SARCASM & SENTIMENT ANALYZER — NLP Demo")
    print("   Layers: Semantics | Pragmatics | Discourse")
    print("=" * 65)
    
    for text, ctx in examples:
        result = analyze(text, ctx)
        print(f"\n📝 Input  : {text}")
        if ctx:
            print(f"💬 Context: {ctx}")
        print(f"📊 Surface: {result.surface_sentiment} ({result.surface_score:+.2f})")
        print(f"🎭 Sarcasm: {'YES' if result.sarcasm_detected else 'NO'} "
              f"(confidence: {result.sarcasm_confidence:.0%})")
        if result.sarcasm_signals:
            for s in result.sarcasm_signals:
                print(f"   ↳ {s}")
        print(f"✅ TRUE   : {result.true_sentiment} ({result.true_score:+.2f})")
        print(f"💡 {result.explanation}")
        print("-" * 65)

import streamlit as st

# 🔥 MUST: paste your entire Python analyzer code ABOVE this (your analyze() function)

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(page_title="Sarcasm Analyzer", layout="centered")

# ─────────────────────────────────────────────
# Custom CSS (from your HTML)
# ─────────────────────────────────────────────
st.markdown("""
<style>
body {
    background-color: #0a0a0f;
    color: #e2e8f0;
}

.main {
    background-color: #0a0a0f;
}

h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 30%, #ff4d6d 70%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

.card {
    background: #16161f;
    border: 1px solid #2a2a3a;
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}

.badge {
    text-align: center;
    color: #ff4d6d;
    font-size: 0.8rem;
}

.label {
    color: #64748b;
    font-size: 0.7rem;
}

.pos { color: #4ade80; }
.neg { color: #f87171; }
.neu { color: #94a3b8; }
.sar { color: #ff4d6d; }

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("<div class='badge'>🧠 NLP Project · 3-Layer Analysis</div>", unsafe_allow_html=True)
st.markdown("<h1>Sarcasm & Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#64748b'>Detects what you really mean — not just what you said.</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Card
# ─────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area("📝 Input Sentence", placeholder="e.g. Great, another exam 😒")

use_context = st.checkbox("💬 Add previous message (context)")
context = None
if use_context:
    context = st.text_area("Previous Message")

col1, col2 = st.columns([1,1])

with col1:
    analyze_btn = st.button("Analyze")

with col2:
    clear_btn = st.button("Clear")

st.markdown("</div>", unsafe_allow_html=True)

# Clear
if clear_btn:
    st.experimental_rerun()

# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────
if analyze_btn and text.strip():
    result = analyze(text, context)

    # Verdict Banner
    if result.sarcasm_detected:
        verdict_color = "#ff4d6d"
        emoji = "🎭"
    elif result.true_sentiment == "Positive":
        verdict_color = "#4ade80"
        emoji = "😊"
    elif result.true_sentiment == "Negative":
        verdict_color = "#f87171"
        emoji = "😞"
    else:
        verdict_color = "#94a3b8"
        emoji = "😐"

    st.markdown(f"""
    <div class='card' style='border:1px solid {verdict_color}'>
        <h3 style='color:{verdict_color}'>{emoji} {result.true_sentiment}</h3>
        <p>"{result.text}"</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Results Grid ──
    col1, col2 = st.columns(2)

    with col1:
        color = "pos" if result.surface_sentiment=="Positive" else "neg" if result.surface_sentiment=="Negative" else "neu"
        st.markdown(f"""
        <div class='card'>
            <div class='label'>Surface Sentiment</div>
            <h2 class='{color}'>{result.surface_sentiment}</h2>
            <p>Score: {result.surface_score}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        color = "pos" if result.true_sentiment=="Positive" else "neg" if result.true_sentiment=="Negative" else "neu"
        st.markdown(f"""
        <div class='card'>
            <div class='label'>True Sentiment</div>
            <h2 class='{color}'>{result.true_sentiment}</h2>
            <p>Score: {result.true_score}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Sarcasm Meter ──
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🎭 Sarcasm Confidence")

    st.progress(result.sarcasm_confidence)

    st.write("Detected:", "YES" if result.sarcasm_detected else "NO")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Explanation ──
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 💡 Explanation")
    st.write(result.explanation)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Signals ──
    if result.sarcasm_signals:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🚨 Sarcasm Signals")
        for s in result.sarcasm_signals:
            st.write("⚡", s)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Breakdown ──
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Layer Breakdown")

    st.write("**Semantics:**", result.layer_breakdown["semantics"]["label"])
    st.write("**Pragmatics:**", "Sarcasm" if result.sarcasm_detected else "Genuine")
    st.write("**Discourse:**", result.layer_breakdown["discourse"]["note"])

    st.markdown("</div>", unsafe_allow_html=True)
