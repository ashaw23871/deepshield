"""
gemini_utils.py — Google Gemini integration for DeepShield.
Uses the new google-genai package (replaces deprecated google-generativeai).
"""

from __future__ import annotations

import os
import logging

logger = logging.getLogger("deepshield.gemini")

try:
    from google import genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed. Gemini explanations disabled.")


def get_gemini_explanation(
    verdict: str,
    score: float,
    breakdown: dict[str, float],
) -> str | None:

    api_key = os.environ.get("GEMINI_API_KEY")

    if not _GEMINI_AVAILABLE:
        return "Gemini explanation unavailable — install 'google-genai' and set GEMINI_API_KEY."

    if not api_key:
        return "Gemini explanation unavailable — set the GEMINI_API_KEY environment variable."

    try:
        client = genai.Client(api_key=api_key)

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

        response = client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
        )
        return response.text.strip()

    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return "AI explanation temporarily unavailable due to high demand. The authenticity score and signal breakdown above provide the full analysis."