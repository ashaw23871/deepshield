"""
gemini_utils.py — Google Gemini integration for DeepShield.
Uses direct REST API to ensure maximum compatibility.
"""

from __future__ import annotations

import os
import logging
import httpx

logger = logging.getLogger("deepshield.gemini")


def get_gemini_explanation(
    verdict: str,
    score: float,
    breakdown: dict[str, float],
) -> str | None:

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return "Gemini explanation unavailable — set the GEMINI_API_KEY environment variable."

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

    models_to_try = [
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    ]

    for model in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }

            with httpx.Client(timeout=30) as client:
                response = client.post(url, json=payload)
                data = response.json()

                if response.status_code == 200:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Gemini success with model: {model}")
                    return text.strip()
                else:
                    logger.warning(f"Model {model} failed: {data.get('error', {}).get('message', '')}")
                    continue

        except Exception as e:
            logger.warning(f"Model {model} exception: {e}")
            continue

    return "Based on our forensic and AI analysis, the authenticity score and signal breakdown above provide a comprehensive assessment. Please refer to the risk level for the recommended course of action."