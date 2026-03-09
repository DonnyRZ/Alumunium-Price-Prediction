from __future__ import annotations


SYSTEM_PROMPT = """
You analyze whether a news article is likely to be bullish, neutral, or bearish for aluminium market prices.

Important:
- Score the likely impact on aluminium price, not whether the article is emotionally positive or negative.
- Supply disruption, tighter inventories, tariffs, sanctions, or force majeure can be bullish for aluminium price.
- Demand weakness, easing tariffs, rising inventories, or weaker premiums can be bearish for aluminium price.
- If the title/snippet is too vague, respond neutral with low confidence.
- Return valid JSON only.
""".strip()


def build_user_prompt(title: str, snippet: str, max_reason_chars: int) -> str:
    snippet = (snippet or "").strip()
    return f"""
Analyze this aluminium-market news article.

Title:
{title}

Snippet:
{snippet if snippet else "N/A"}

Return valid JSON with this exact structure:
{{
  "market_impact_score": -1.0,
  "impact_label": "bearish|neutral|bullish",
  "impact_channel": "price|supply|policy|logistics|inventory|demand|macro|unclear",
  "confidence": 0.0,
  "reason_short": "short explanation"
}}

Rules:
1. `market_impact_score` is from -1 to 1.
2. Negative means bearish for aluminium price.
3. Positive means bullish for aluminium price.
4. `confidence` is from 0 to 1.
5. `reason_short` must be concise and no more than about {max_reason_chars} characters.
6. If evidence is weak or the article is too broad, choose `neutral` with lower confidence.
""".strip()
