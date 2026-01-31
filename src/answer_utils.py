import re

MAX_ANSWER_CHARS = 89
FALLBACK_ANSWER = "idk"

_NUMERIC_HINTS = [
    "arabic numerals",
    "numerals",
    "digits",
    "provide in arabic",
]

def is_numeric_question(q: str) -> bool:
    ql = (q or "").lower()
    if any(h in ql for h in _NUMERIC_HINTS):
        return True
    if ql.strip().startswith("how many"):
        return True
    # sometimes "Provide in Arabic numerals (e.g., 7, 8) only."
    if "e.g." in ql and "only" in ql and any(ch.isdigit() for ch in ql):
        return True
    return False

def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()

    # take first line
    s = s.splitlines()[0].strip()

    # strip common prefixes
    s = re.sub(r"^\s*(answer\s*[:\-]\s*)", "", s, flags=re.IGNORECASE).strip()

    # remove quotes
    s = s.strip(" \"'`")

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    # remove trailing punctuation
    s = re.sub(r"[.?!,:;]+$", "", s).strip()

    # if still long, hard cap
    if len(s) > MAX_ANSWER_CHARS:
        s = s[:MAX_ANSWER_CHARS].rstrip()

    return s

def normalize_for_exact_match(ans: str) -> str:
    # aggressive normalization for voting / matching style
    s = normalize_answer(ans).lower()
    # remove surrounding punctuation
    s = s.strip(" .,:;!?\"'`")
    # collapse spaces again
    s = re.sub(r"\s+", " ", s).strip()
    return s

def enforce_numeric(ans: str) -> str:
    """Keep only a single integer if present; else empty."""
    if ans is None:
        return ""
    m = re.search(r"\d+", str(ans))
    return m.group(0) if m else ""

def final_postprocess(ans: str, question: str) -> str:
    """Apply numeric rules + fallback + length cap."""
    if is_numeric_question(question):
        a = enforce_numeric(ans)
        if not a:
            a = FALLBACK_ANSWER
        return a

    a = normalize_answer(ans)
    if not a.strip():
        a = FALLBACK_ANSWER
    return a
