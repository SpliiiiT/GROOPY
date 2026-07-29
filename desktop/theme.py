"""Shared visual theme for the GROOPY desktop apps (PyQt5 QSS).

One dark, robust look applied across the launcher, recognition, and synthesis windows so the
whole app feels like a single product. Palette matches the project's docs/artifacts (teal
accent on a cool near-black). Import and call apply_theme(app) once per QApplication.
"""
from __future__ import annotations

# --- palette (kept in sync with the docs artifacts) --------------------------------------
BG = "#0e141a"          # window background
SURFACE = "#161f27"     # cards / inputs
SURFACE2 = "#1c2831"    # hover / secondary
BORDER = "#27333d"
INK = "#e7edf2"         # primary text
MUTED = "#93a2b0"       # secondary text
ACCENT = "#17b8a6"      # teal
ACCENT_HOVER = "#2ed6c1"
ACCENT_DK = "#0f8b7e"
GOOD = "#2fae82"        # positive sentiment
WARN = "#d9694f"        # negative sentiment
NEUTRAL = "#7f8c99"     # neutral sentiment

QSS = f"""
QWidget {{
    background: {BG};
    color: {INK};
    font-family: "Segoe UI", system-ui, sans-serif;
    font-size: 14px;
}}
QMainWindow, QDialog {{ background: {BG}; }}

/* section / card frames (objectName='card') */
QFrame#card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}

QLabel {{ background: transparent; }}
QLabel#title    {{ font-size: 22px; font-weight: 700; color: {INK}; }}
QLabel#subtitle {{ font-size: 13px; color: {MUTED}; }}
QLabel#pred     {{ font-size: 26px; font-weight: 700; color: {ACCENT_HOVER}; }}
QLabel#sentence {{ font-size: 20px; color: {INK}; }}
QLabel#section  {{ font-size: 11px; font-weight: 700; color: {MUTED};
                   letter-spacing: 2px; text-transform: uppercase; }}
QLabel#status   {{ font-size: 12px; color: {MUTED}; }}

/* primary buttons */
QPushButton {{
    background: {ACCENT};
    color: {BG};
    border: none;
    border-radius: 9px;
    padding: 10px 16px;
    font-weight: 600;
    font-size: 14px;
}}
QPushButton:hover  {{ background: {ACCENT_HOVER}; }}
QPushButton:pressed {{ background: {ACCENT_DK}; }}
QPushButton:disabled {{ background: {SURFACE2}; color: {MUTED}; }}

/* secondary buttons (objectName='ghost') */
QPushButton#ghost {{
    background: {SURFACE2};
    color: {INK};
    border: 1px solid {BORDER};
}}
QPushButton#ghost:hover {{ border-color: {ACCENT}; color: {ACCENT_HOVER}; }}

/* big primary CTA (launcher) */
QPushButton#cta {{ font-size: 16px; padding: 16px 18px; border-radius: 12px; text-align: left; }}

QLineEdit, QTextEdit {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 9px;
    padding: 10px 12px;
    color: {INK};
    font-size: 15px;
    selection-background-color: {ACCENT_DK};
}}
QLineEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}

QCheckBox {{ color: {MUTED}; spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
                        border: 1px solid {BORDER}; background: {SURFACE}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
"""


def apply_theme(app) -> None:
    """Apply the GROOPY dark theme to a QApplication."""
    app.setStyleSheet(QSS)


def sentiment_style(sentiment) -> str:
    """Inline stylesheet for a sentiment 'badge' QLabel, coloured by label."""
    tone = getattr(sentiment, "label", "neutral") if sentiment else "neutral"
    color = {"positive": GOOD, "negative": WARN, "neutral": NEUTRAL}.get(tone, NEUTRAL)
    return (f"background: {color}; color: {BG}; border-radius: 11px; "
            f"padding: 4px 12px; font-weight: 700; font-size: 13px;")


def set_sentiment_badge(label, sentiment) -> None:
    """Update a QLabel to show a coloured sentiment pill (or hide it if no sentiment)."""
    if sentiment is None:
        label.setText("")
        label.setVisible(False)
        return
    label.setVisible(True)
    label.setText(f"  {sentiment.label.upper()}  ·  {sentiment.score:.0%}  ")
    label.setStyleSheet(sentiment_style(sentiment))
