"""
gemini_utils.py — Google Gemini integration for DeepShield.

Uses the Gemini 1.5 Flash model to generate a concise, human-readable
explanation of the deepfake analysis result. This satisfies the Google
Solution Challenge requirement to integrate a Google AI product.

Setup:
    pip install google-generativeai
    export GEMINI_API_KEY="your-key-here"

Get a free key at: https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger("deepshield.gemini")

# Lazy import so the app still runs if the package isn't installed
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Gemini explanations disabled.")


def get_gemini_explanation(
    verdict: str,
    score: float,
    breakdown: dict[str, float],
) -> str | None:
    """
    Call Gemini 1.5 Flash to generate a plain-English explanation of the
    analysis result, tailored for sports journalists and media consumers.

    Returns None gracefully if the API key is missing or the call fails,
    so the rest of the analysis is never blocked.
    """
    api_key = os.environ.get("GEMINI_API_KEY")

    if not _GEMINI_AVAILABLE:
        return (
            "Gemini explanation unavailable — install 'google-generativeai' "
            "and set GEMINI_API_KEY to enable this feature."
        )

    if not api_key:
        return (
            "Gemini explanation unavailable — set the GEMINI_API_KEY "
            "environment variable to enable this feature."
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        breakdown_text = "\n".join(
            f"  - {k.replace('_', ' ').title()}: {v}/100"
            for k, v in breakdown.items()
        )

        prompt = f"""You are DeepShield, an AI assistant that helps sports journalists and 
fans understand whether a sports video or image has been digitally manipulated.

A piece of sports media has just been analyzed. Here are the results:

Verdict: {verdict}
Overall Authenticity Score: {score:.1f}/100
Score Breakdown:
{breakdown_text}

Write a clear, confident, 2-3 sentence explanation of what these results mean 
for a non-technical sports journalist. Mention the most significant signal(s) 
that drove the verdict. Use plain English — no jargon. Do not mention specific 
model names or internal variable names. End with a practical recommendation 
(e.g., "We recommend seeking original broadcast footage before publishing.")"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return f"Gemini explanation temporarily unavailable: {type(e).__name__}"
