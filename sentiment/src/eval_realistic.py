"""Second sentiment evaluation pass — realistic, domain-matched text.

The bake-off (evaluate.py) scores on IMDB movie reviews, which has two problems for THIS
app: (1) IMDB has no neutral examples, so it can't reward/penalise neutral-detection quality,
and (2) movie reviews are a different register than the short typed/spoken utterances this app
actually handles (greetings, simple statements, everyday emotion). This is a small, hand-labeled
set in that actual register, used to sanity-check the bake-off's pick against realistic input
rather than trust IMDB accuracy alone.

Usage:
  python -m sentiment.src.eval_realistic
"""
from __future__ import annotations

from .models import REGISTRY

# (text, true_label) — greetings/statements/simple emotion/negation, the kind of thing typed
# or spoken into the synthesis app, not movie-review prose.
EXAMPLES = [
    ("hello my name is oussama", "neutral"),
    ("the box is on the table", "neutral"),
    ("I am going to the store", "neutral"),
    ("can you help me with this", "neutral"),
    ("the meeting is at three o clock", "neutral"),
    ("she is reading a book", "neutral"),
    ("thank you so much, I really appreciate it", "positive"),
    ("I am so happy to see you today", "positive"),
    ("this is wonderful, great job everyone", "positive"),
    ("I love spending time with my family", "positive"),
    ("what a fantastic surprise, thank you", "positive"),
    ("I am excited for the presentation", "positive"),
    ("I am so sad and tired today", "negative"),
    ("this is terrible, I hate it", "negative"),
    ("I am sorry, that was a mistake", "negative"),
    ("this hurts, I am in pain", "negative"),
    ("I am upset and angry about this", "negative"),
    ("that was the worst experience ever", "negative"),
    ("not good, I am not happy with this", "negative"),  # negation
    ("I don't like this at all", "negative"),  # negation
]


def main() -> None:
    print(f"{len(EXAMPLES)} hand-labeled examples: "
          f"{sum(1 for _, l in EXAMPLES if l == 'neutral')} neutral / "
          f"{sum(1 for _, l in EXAMPLES if l == 'positive')} positive / "
          f"{sum(1 for _, l in EXAMPLES if l == 'negative')} negative\n")

    for name, build_fn in REGISTRY.items():
        backend = build_fn()
        correct = 0
        by_class = {"positive": [0, 0], "negative": [0, 0], "neutral": [0, 0]}  # [correct, total]
        for text, true_label in EXAMPLES:
            pred = backend.score(text).label
            by_class[true_label][1] += 1
            if pred == true_label:
                correct += 1
                by_class[true_label][0] += 1
        acc = correct / len(EXAMPLES)
        breakdown = ", ".join(f"{c}={n[0]}/{n[1]}" for c, n in by_class.items())
        print(f"[{name}] overall acc={acc:.2f}  ({breakdown})")


if __name__ == "__main__":
    main()
