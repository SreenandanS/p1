import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import json
import re
from typing import Dict, Any

load_dotenv()

DATA_FOLDER = "data-source"
OUTPUT_FOLDER = "./final_reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

try:
    # Pathway LLM is optional; we'll attempt to import and use it if available and API key is set
    import pathway as pw
    from pathway.xpacks.llm import llms
except Exception:
    pw = None
    llms = None


EXPECTED_FILES = [
    "bear_report.md",
    "bull_report.md",
    "fundamentals_report.md",
    "market_report.md",
    "news_report.md",
    "sentiment_report.md",
]


def load_reports(data_folder: str = DATA_FOLDER) -> Dict[str, str]:
    """Read expected markdown reports from data folder and return a mapping.

    Raises FileNotFoundError if any required file is missing.
    """
    reports = {}
    missing = []
    for fname in EXPECTED_FILES:
        path = Path(data_folder) / fname
        if not path.exists():
            missing.append(fname)
        else:
            reports[fname] = path.read_text()

    if missing:
        raise FileNotFoundError(f"Missing files in {data_folder}: {', '.join(missing)}")

    return reports


def simple_sentiment_score(text: str) -> float:
    """Very small heuristic sentiment score based on keyword counts.

    Returns score in [-1.0, 1.0] where positive => bullish sentiment, negative => bearish.
    """
    if not text:
        return 0.0

    text_l = text.lower()
    positive = len(re.findall(r"\b(good|strong|positive|bullish|growth|beat|outperform|optimis)\w*\b", text_l))
    negative = len(re.findall(r"\b(bad|weak|negative|bearish|risk|concern|downturn|underperform|decline|recession)\w*\b", text_l))
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total


def heuristic_risk_assessment(reports: Dict[str, str]) -> Dict[str, Any]:
    """Create a fast heuristic risk assessment from available reports.

    This is a fallback that does not require an LLM. It produces a structured dict
    describing overall risk level and basic position-sizing recommendations.
    """
    sentiment = simple_sentiment_score(reports.get("sentiment_report.md", ""))
    market_sentiment = simple_sentiment_score(reports.get("market_report.md", ""))
    fundamentals_sentiment = simple_sentiment_score(reports.get("fundamentals_report.md", ""))

    # Combine signals with simple weights
    combined = 0.5 * sentiment + 0.3 * fundamentals_sentiment + 0.2 * market_sentiment

    # Map to risk levels
    if combined >= 0.3:
        risk_level = "Low"
        position_pct = 0.10  # 10% of portfolio
        stop_loss = 0.12
    elif combined <= -0.3:
        risk_level = "High"
        position_pct = 0.02  # 2% of portfolio
        stop_loss = 0.06
    else:
        risk_level = "Medium"
        position_pct = 0.05
        stop_loss = 0.08

    return {
        "method": "heuristic",
        "combined_signal": combined,
        "risk_level": risk_level,
        "recommended_position_pct": position_pct,
        "suggested_stop_loss_pct": stop_loss,
        "notes": "Heuristic fallback based on keyword sentiment counts from reports."
    }


def llm_risk_assessment(reports: Dict[str, str], portfolio_value: float = 100000.0) -> Dict[str, Any]:
    """Call an LLM (via Pathway llms wrapper) to produce a structured risk assessment.

    If Pathway or API key is not available, raises RuntimeError.
    The LLM is asked to return a JSON object with keys:
      overall_risk, position_pct, stop_loss_pct, hedge_recommendation, rationale
    """
    if llms is None or pw is None:
        raise RuntimeError("Pathway or llms not available in this environment")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot call LLM")

    chat_model = llms.OpenAIChat(model="gpt-4o-mini", temperature=0.3, api_key=api_key)

    # Build system and user prompts
    system_prompt = (
        "You are a Risk Management Team assistant. Given the provided research reports,\n"
        "produce a concise, structured JSON with risk guidance for a single trade.\n"
        "The JSON must have these fields: overall_risk (Low/Medium/High), position_pct (0-1), "
        "stop_loss_pct (0-1), hedge_recommendation (string), rationale (string).\n"
        "Respond ONLY with valid JSON. No additional text."
    )

    combined_text = (
        "Bull Report:\n" + reports.get("bull_report.md", "")[:3000]
        + "\n\nBear Report:\n" + reports.get("bear_report.md", "")[:3000]
        + "\n\nMarket:\n" + reports.get("market_report.md", "")[:2000]
        + "\n\nFundamentals:\n" + reports.get("fundamentals_report.md", "")[:2000]
        + "\n\nNews:\n" + reports.get("news_report.md", "")[:2000]
        + "\n\nSentiment:\n" + reports.get("sentiment_report.md", "")[:2000]
    )

    user_prompt = (
        f"Portfolio value: {portfolio_value}\n\n"
        "Available research (truncated):\n"
        + combined_text
        + "\n\nPlease produce the JSON risk guidance now."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Use Pathway debug table flow similar to other repo files so this integrates well
    table = pw.debug.table_from_pandas(
        __import__("pandas").DataFrame({"messages": [messages]})
    )
    response = table.select(risk=chat_model(pw.this.messages))
    result = pw.debug.table_to_pandas(response)
    risk_json_text = result["risk"].iloc[0] if not result.empty else None

    if not risk_json_text:
        raise RuntimeError("LLM returned empty response")

    # Try to parse JSON from the model's reply
    try:
        parsed = json.loads(risk_json_text)
    except Exception:
        # If the model returned text with code fences or extra text, try to extract JSON block
        m = re.search(r"\{[\s\S]*\}", str(risk_json_text))
        if m:
            parsed = json.loads(m.group(0))
        else:
            raise

    # Normalize fields
    parsed.setdefault("method", "llm")
    return parsed


def assess_risk(reports: Dict[str, str], portfolio_value: float = 100000.0) -> Dict[str, Any]:
    """High-level API: try LLM assessment first, fall back to heuristic if unavailable."""
    try:
        assessment = llm_risk_assessment(reports, portfolio_value=portfolio_value)
        assessment["source"] = "llm"
        return assessment
    except Exception as e:
        # Do not fail hard — return heuristic assessment with note
        h = heuristic_risk_assessment(reports)
        h["source"] = "heuristic"
        h["llm_error"] = str(e)
        return h


def write_reports(assessment: Dict[str, Any], output_folder: str = OUTPUT_FOLDER) -> Path:
    """Write both machine-readable JSON and a human-readable summary to `output_folder`.

    Returns the path to the JSON file created.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = Path(output_folder) / f"risk_management_report_{timestamp}.json"
    txt_path = Path(output_folder) / f"risk_management_report_{timestamp}.txt"

    json_path.write_text(json.dumps(assessment, indent=2))

    # Human readable
    lines = [
        "Risk Management Report",
        f"Generated: {timestamp}",
        "",
        f"Source: {assessment.get('source')}",
        f"Overall risk: {assessment.get('risk_level') or assessment.get('overall_risk')}",
        f"Recommended position (% of portfolio): {assessment.get('recommended_position_pct') or assessment.get('position_pct')}",
        f"Suggested stop-loss (%): {assessment.get('suggested_stop_loss_pct') or assessment.get('stop_loss_pct')}",
        "",
        "Rationale:",
        assessment.get('notes') or assessment.get('rationale', ''),
    ]

    txt_path.write_text("\n".join(lines))
    return json_path


if __name__ == "__main__":
    try:
        reports = load_reports()
    except Exception as e:
        print(f"Error loading reports: {e}")
        raise

    assessment = assess_risk(reports, portfolio_value=100000.0)
    out = write_reports(assessment)
    print(f"Risk report written to: {out}")
