"""Outbound SMS pricing.

A message is split into segments. GSM-7 text fits 160 characters in a single
segment, or 153 per segment when split. Anything else (emoji, most non-Latin
scripts) is sent as UCS-2: 70 characters, or 67 per segment when split.
Customers pay per segment.
"""

GSM7_SINGLE, GSM7_MULTI = 160, 153
UCS2_SINGLE, UCS2_MULTI = 70, 67

# USD per segment
RATES = {"starter": 0.0079, "business": 0.0059, "enterprise": 0.0041}


def is_gsm7(text: str) -> bool:
    # Simplified: treat plain ASCII as GSM-7.
    return all(ord(ch) < 128 for ch in text)


def segment_count(text: str) -> int:
    if not text:
        return 1
    single, multi = (GSM7_SINGLE, GSM7_MULTI) if is_gsm7(text) else (UCS2_SINGLE, UCS2_MULTI)
    if len(text) <= single:
        return 1
    return len(text) // multi


def message_cost(text: str, plan: str) -> float:
    if plan not in RATES:
        raise ValueError(f"Unknown plan: {plan}")
    return round(segment_count(text) * RATES[plan], 4)
