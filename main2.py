import pathway as pw
import os
from pathlib import Path
from dotenv import load_dotenv
from pathway.xpacks.llm import llms
import json
from datetime import datetime
import time
import pandas as pd

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = "data-source"
OUTPUT_FOLDER = "./final_reports"
N_ROUNDS = 3  
load_dotenv()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# MODEL SETUP
# ----------------------------
chat_model = llms.OpenAIChat(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ----------------------------
# SYSTEM PROMPTS
# ----------------------------
BEAR_SYSTEM_PROMPT = """You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators. Leverage the provided research and data to highlight potential downsides and counter bullish arguments effectively.

Key points to focus on:

- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats that could hinder the stock's performance.
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning, declining innovation, or threats from competitors.
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news to support your position.
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning, exposing weaknesses or over-optimistic assumptions.
- Engagement: Present your argument in a conversational style, directly engaging with the bull analyst's points and debating effectively rather than simply listing facts.

Use this information to deliver a compelling bear argument, refute the bull's claims, and engage in a dynamic debate that demonstrates the risks and weaknesses of investing in the stock. You must also address reflections and learn from lessons and mistakes you made in the past."""

BULL_SYSTEM_PROMPT = """You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators. Leverage the provided research and data to address concerns and counter bearish arguments effectively.

Key points to focus on:
- Growth Potential: Highlight the company's market opportunities, revenue projections, and scalability.
- Competitive Advantages: Emphasize factors like unique products, strong branding, or dominant market positioning.
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence.
- Bear Counterpoints: Critically analyze the bear argument with specific data and sound reasoning, addressing concerns thoroughly and showing why the bull perspective holds stronger merit.
- Engagement: Present your argument in a conversational style, engaging directly with the bear analyst's points and debating effectively rather than just listing data.

Use this information to deliver a compelling bull argument, refute the bear's concerns, and engage in a dynamic debate that demonstrates the strengths of the bull position. You must also address reflections and learn from lessons and mistakes you made in the past."""

SUMMARIZER_SYSTEM_PROMPT = """You are a summarizer AI. After reading the debate between Bull and Bear agents, summarize the discussion objectively and conclude with a clear recommendation: BUY, HOLD, or SELL."""

# ----------------------------
# DEBATE EXECUTION
# ----------------------------
def execute_debate(fundamentals: str, market: str, news: str, sentiment: str) -> str:
    """Execute the complete debate with all reports"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"🔔 NEW DEBATE TRIGGERED at {timestamp}")
    print(f"{'='*60}")
    
    # Initialize debate
    history = []
    bear_message = "Let's begin. I believe there are significant risks investors should be aware of."
    
    # Run debate rounds
    for round_num in range(1, N_ROUNDS + 1):
        print(f"\n📍 Round {round_num}")
        print("-" * 50)
        
        # Bull's turn
        print("🐂 Bull Analyst thinking...")
        bull_prompt = [
            {"role": "system", "content": BULL_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Market Research Report:
{market}

Social Media Sentiment Report:
{sentiment}

World Affairs News:
{news}

Company Fundamentals Report:
{fundamentals}

Conversation history of the debate:
{json.dumps(history)}

Round {round_num}:
Last bear argument: {bear_message}

Your turn to argue as the Bull Analyst. Provide a compelling, well-reasoned response that directly engages with the bear analyst's points and demonstrates the strengths of investing in this stock."""}
        ]
        
        # Create table and process with LLM
        bull_table = pw.debug.table_from_pandas(
            pd.DataFrame({"messages": [bull_prompt]})
        )
        bull_response = bull_table.select(reply=chat_model(pw.this.messages))
        bull_result = pw.debug.table_to_pandas(bull_response)
        bull_reply = bull_result["reply"].iloc[0] if not bull_result.empty else f"Bull argues for round {round_num}"
        
        print(f"Bull: {bull_reply[:150]}...")
        
        # Bear's turn
        print("🐻 Bear Analyst thinking...")
        bear_prompt = [
            {"role": "system", "content": BEAR_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Market Research Report:
{market}

Social Media Sentiment Report:
{sentiment}

World Affairs News:
{news}

Company Fundamentals Report:
{fundamentals}

Conversation history of the debate:
{json.dumps(history)}

Round {round_num}:
Last bull argument: {bull_reply}

Your turn to argue as the Bear Analyst. Respond critically with a compelling argument that highlights risks and challenges, and directly engages with the bull analyst's points."""}
        ]
        
        bear_table = pw.debug.table_from_pandas(
            pd.DataFrame({"messages": [bear_prompt]})
        )
        bear_response = bear_table.select(reply=chat_model(pw.this.messages))
        bear_result = pw.debug.table_to_pandas(bear_response)
        bear_reply = bear_result["reply"].iloc[0] if not bear_result.empty else f"Bear counters for round {round_num}"
        
        print(f"Bear: {bear_reply[:150]}...")
        
        history.append({"round": round_num, "bull": bull_reply, "bear": bear_reply})
        bear_message = bear_reply
    
    # Generate summary
    print("\n📊 Generating final summary...")
    debate_text = "\n\n".join([
        f"Round {item['round']}:\nBull: {item['bull']}\nBear: {item['bear']}" 
        for item in history
    ])
    
    summary_prompt = [
        {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
        {"role": "user", "content": f"""You are an objective financial analyst. Review the following debate between Bull and Bear analysts and provide a comprehensive summary with a clear recommendation.

Available Research:

Market Research Report:
{market}

Social Media Sentiment Report:
{sentiment}

World Affairs News:
{news}

Company Fundamentals Report:
{fundamentals}

Debate Transcript:
{debate_text}

Provide:
1. A balanced summary of key arguments from both sides
2. An assessment of the strongest points made
3. A clear final recommendation: BUY, HOLD, or SELL
4. Reasoning for your recommendation based on the debate and available data

Be objective and consider both perspectives before concluding."""}
    ]
    
    summary_table = pw.debug.table_from_pandas(
        pd.DataFrame({"messages": [summary_prompt]})
    )
    summary_response = summary_table.select(summary=chat_model(pw.this.messages))
    summary_result = pw.debug.table_to_pandas(summary_response)
    summary = summary_result["summary"].iloc[0] if not summary_result.empty else "Summary completed"
    
    # Save to file
    clean_timestamp = timestamp.replace(':', '-').replace(' ', '_')
    output_path = Path(OUTPUT_FOLDER) / f"final_summary_{clean_timestamp}.txt"
    full_report = f"""Stock Analysis Debate Summary
Generated: {timestamp}

{'='*60}
FINAL SUMMARY AND RECOMMENDATION
{'='*60}

{summary}

{'='*60}
DEBATE TRANSCRIPT
{'='*60}

{debate_text}
"""
    output_path.write_text(full_report)
    
    print(f"\n✅ Report saved: {output_path.name}")
    print(f"{'='*60}\n")
    
    return summary

# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    print("🚀 Pathway Stock Debate System")
    print("=" * 60)
    
    # Check for required files
    txt_files = sorted(Path(DATA_FOLDER).glob("*.md"))
    if len(txt_files) < 4:
        raise ValueError(
            f"Expected at least 4 markdown files in '{DATA_FOLDER}' folder. "
            f"Found {len(txt_files)}. Please ensure you have:\n"
            "1. fundamentals.md\n"
            "2. market_research.md\n"
            "3. news.md\n"
            "4. sentiment.md"
        )
    
    print(f"📁 Found {len(txt_files)} report files")
    print(f"📊 Loading reports from {DATA_FOLDER}/")
    print("⚡ Streaming mode with file monitoring")
    print("=" * 60)
    
    # Run initial debate
    reports = [Path(f).read_text() for f in txt_files[:4]]
    fundamentals, market, news, sentiment = reports
    summary = execute_debate(fundamentals, market, news, sentiment)
    
    print("\n" + "="*60)
    print("INITIAL SUMMARY")
    print("="*60)
    print(summary)
    print("="*60)
    
    # Monitor for file changes
    print("\n💡 System is now monitoring for file changes...")
    print("🛑 Press Ctrl+C to stop\n")
    
    last_modified_times = {f: f.stat().st_mtime for f in txt_files[:4]}
    
    try:
        while True:
            time.sleep(60)  # Check every 60 seconds
            
            # Check if any file has been modified
            for file_path in txt_files[:4]:
                current_mtime = file_path.stat().st_mtime
                if current_mtime != last_modified_times[file_path]:
                    print(f"\n🔔 File change detected: {file_path.name}")
                    print("🔄 Re-running debate...")
                    
                    # Update all modification times
                    last_modified_times = {f: f.stat().st_mtime for f in txt_files[:4]}
                    
                    # Re-read all files and run debate
                    new_reports = [Path(f).read_text() for f in txt_files[:4]]
                    summary = execute_debate(*new_reports)
                    
                    print("\n" + "="*60)
                    print("UPDATED SUMMARY")
                    print("="*60)
                    print(summary)
                    print("="*60)
                    
                    break  # Exit the for loop to restart monitoring
                    
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user")
        print("✅ System shutdown complete")