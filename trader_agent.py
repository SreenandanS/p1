import pathway as pw
from pathway.xpacks.llm import llms
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

load_dotenv()

# === Config ===
DATA_FOLDER = "data-source"
OUTPUT_FOLDER = "./final_reports"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

files = [
    "bear_report.md",
    "bull_report.md",
    "fundamentals_report.md",
    "news_report.md",
    "market_report.md",
    "sentiment_report.md",
]

# ✅ Check if all required files exist
missing_files = [f for f in files if not os.path.exists(os.path.join(DATA_FOLDER, f))]
if missing_files:
    raise FileNotFoundError(f"Missing files: {', '.join(missing_files)}")


chat_model = llms.OpenAIChat(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
)

TRADER_SYSTEM_PROMPT = """
You are a Trader Agent responsible for making the final investment decision after reviewing 
detailed analyses from both the Bull and Bear Analysts. Your role is to synthesize their arguments, 
weigh the supporting evidence, and arrive at a balanced, data-driven trading decision.

Your decision-making process should:
- Objectively evaluate both perspectives without bias.
- Consider macroeconomic indicators, sentiment data, and market research provided.
- Identify the strongest points from each analyst and determine which carries more weight.
- Address conflicting reasoning logically and justify your final stance with clear rationale.
- Prioritize capital preservation, risk-adjusted returns, and long-term sustainability.

Structure your analysis as follows:
1. **Summary of Bull Position:** Briefly restate key bullish arguments.
2. **Summary of Bear Position:** Briefly restate key bearish arguments.
3. **Comparative Evaluation:** Analyze where the two positions diverge and which is better supported by data.
4. **Final Reasoning:** Provide a concise explanation of your final view.
5. **Decision:** Conclude with a clear recommendation.

Always end your response with:
'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**'
to confirm your final investment decision.
"""

# === Read markdown reports using Pathway ===
bear_report = pw.io.fs.read(f"{DATA_FOLDER}/bear_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300, with_metadata=True)
bull_report = pw.io.fs.read(f"{DATA_FOLDER}/bull_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300, with_metadata=True)
fundamental_report = pw.io.fs.read(f"{DATA_FOLDER}/fundamental_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300, with_metadata=True)
news_report = pw.io.fs.read(f"{DATA_FOLDER}/news_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300,with_metadata=True)
market_report = pw.io.fs.read(f"{DATA_FOLDER}/market_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300, with_metadata=True)
sentiment_report = pw.io.fs.read(f"{DATA_FOLDER}/sentiment_report.md",mode="streaming", format="plaintext_by_file", autocommit_duration_ms=300, with_metadata=True)

@pw.udf
def combine_reports(market, sentiment, news, fundamentals):
    return f"{market}\n\n{sentiment}\n\n{news}\n\n{fundamentals}"

curr_situation = combine_reports(market_report, sentiment_report, news_report, fundamental_report)

trader_prompt = [
    {"role": "system", "content": TRADER_SYSTEM_PROMPT},
    {"role": "system", "content": f"""Bull Report:
        {bull_report}

        Bear Report:
        {bear_report}

        Market Situation:
        {curr_situation}
    """}
]

# Create a Pathway table with the message
trader_table = pw.debug.table_from_pandas(
    pd.DataFrame({"messages": [trader_prompt]})
)

# Pass through the model — this is the key Pathway call
trader_response = trader_table.select(reply=chat_model(pw.this.messages))

# Convert back to pandas so you can print / inspect
trader_result = pw.debug.table_to_pandas(trader_response)
trader_reply = trader_result["reply"].iloc[0] if not trader_result.empty else ""

# === Write results out ===
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
output_path = Path(OUTPUT_FOLDER) / f"Trader_agent.txt"
full_report = f"""Trader agent Analysis 
    Generated: {timestamp}
   {trader_reply}
    """

output_path.write_text(full_report)

if __name__ == "__main__":
    print("="*80)
    print("🚀 PATHWAY Trader Agent - FULL INTEGRATION")
    print("="*80)
    pw.run()
