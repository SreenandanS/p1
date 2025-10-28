import pathway as pw
import os
from pathlib import Path
from dotenv import load_dotenv
from pathway.xpacks.llm import llms
import json
from datetime import datetime
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
# PATHWAY UDF FOR LLM CALLS
# ----------------------------
def execute_debate_round(fundamentals, market, news, sentiment, history, bear_message, round_num):
    """Execute a single debate round using Pathway"""
    # Bull's turn
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

Your turn to argue as the Bull Analyst."""}
    ]
    
    bull_table = pw.debug.table_from_pandas(pd.DataFrame({"messages": [bull_prompt]}))
    bull_response = bull_table.select(reply=chat_model(pw.this.messages))
    bull_result = pw.debug.table_to_pandas(bull_response)
    bull_reply = bull_result["reply"].iloc[0] if not bull_result.empty else ""
    
    # Bear's turn
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

Your turn to argue as the Bear Analyst."""}
    ]
    
    bear_table = pw.debug.table_from_pandas(pd.DataFrame({"messages": [bear_prompt]}))
    bear_response = bear_table.select(reply=chat_model(pw.this.messages))
    bear_result = pw.debug.table_to_pandas(bear_response)
    bear_reply = bear_result["reply"].iloc[0] if not bear_result.empty else ""
    
    return bull_reply, bear_reply

# ----------------------------
# PATHWAY TRANSFORMATION FUNCTIONS
# ----------------------------
@pw.udf
def extract_filename(path: str) -> str:
    """Extract filename from path"""
    return Path(path).name if path else "unknown"

@pw.udf
def compute_hash(data: str) -> str:
    """Compute hash of file content to detect actual changes"""
    import hashlib
    return hashlib.md5(data.encode()).hexdigest() if data else ""

@pw.udf
def has_four_files(files_tuple) -> bool:
    """Check if we have exactly 4 files"""
    return len(files_tuple) == 4 if files_tuple else False

@pw.udf
def process_debate_data(fundamentals: str, market: str, news: str, sentiment: str) -> str:
    """Main debate processing function that runs in Pathway pipeline"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*80}")
    print(f"🔔 DEBATE TRIGGERED at {timestamp}")
    print(f"{'='*80}")
    
    try:
        # Execute debate
        history = []
        bear_message = "Let's begin. I believe there are significant risks investors should be aware of."
        
        for round_num in range(1, N_ROUNDS + 1):
            print(f"\n📍 Round {round_num}")
            print("-" * 50)
            print("🐂 Bull Analyst thinking...")
            print("🐻 Bear Analyst thinking...")
            
            bull_reply, bear_reply = execute_debate_round(
                fundamentals, market, news, sentiment, 
                history, bear_message, round_num
            )
            
            print(f"Bull: {bull_reply[:100]}...")
            print(f"Bear: {bear_reply[:100]}...")
            
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
    Market Research Report: {market[:200]}...
    Social Media Sentiment Report: {sentiment[:200]}...
    World Affairs News: {news[:200]}...
    Company Fundamentals Report: {fundamentals[:200]}...

    Debate Transcript:
    {debate_text}

    Provide:
    1. A balanced summary of key arguments from both sides
    2. An assessment of the strongest points made
    3. A clear final recommendation: BUY, HOLD, or SELL
    4. Reasoning for your recommendation based on the debate and available data

    Be objective and consider both perspectives before concluding."""}
            ]
            
            summary_table = pw.debug.table_from_pandas(pd.DataFrame({"messages": [summary_prompt]}))
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
        print(f"{'='*80}\n")
        
        return summary
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

# ----------------------------
# MAIN PATHWAY PIPELINE
# ----------------------------
if __name__ == "__main__":
    print("="*80)
    print("🚀 PATHWAY STOCK DEBATE SYSTEM - FULL INTEGRATION")
    print("="*80)
    
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
    print(f"📊 Setting up Pathway streaming pipeline")
    print("="*80)
    
    # PATHWAY STREAMING INPUT - Read files with streaming mode
    files = pw.io.fs.read(
        path=f"{DATA_FOLDER}/*.md",
        format="plaintext_by_file",
        mode="streaming",
        autocommit_duration_ms=3000,  # Check every 3 seconds
        with_metadata=True
    )
    
    # PATHWAY TRANSFORMATION - Add content hash for change detection
    files = files.select(
        pw.this.data,
        content_hash=compute_hash(pw.this.data)
    )
    
    # PATHWAY GROUPBY - Group all files together by a constant key
    # This ensures we process all 4 files together, not individually
    files_with_key = files.select(
        pw.this.data,
        pw.this.content_hash,
        group_key=pw.apply(lambda x: "all_files", pw.this.data)
    )
    
    # PATHWAY REDUCE - Collect all file data together
    # Using reducers to aggregate the files
    grouped = files_with_key.groupby(pw.this.group_key).reduce(
        pw.this.group_key,
        files_data=pw.reducers.tuple(pw.this.data),
        hashes=pw.reducers.tuple(pw.this.content_hash)
    )
    
    # PATHWAY FILTER - Only process when we have exactly 4 files
    grouped = grouped.filter(has_four_files(pw.this.files_data))
    
    # PATHWAY SELECT - Extract individual reports and run debate
    # Map the files to their expected order: fundamentals, market, news, sentiment
    results = grouped.select(
        fundamentals=pw.apply(lambda files: files[0] if len(files) > 0 else "", pw.this.files_data),
        market=pw.apply(lambda files: files[1] if len(files) > 1 else "", pw.this.files_data),
        news=pw.apply(lambda files: files[2] if len(files) > 2 else "", pw.this.files_data),
        sentiment=pw.apply(lambda files: files[3] if len(files) > 3 else "", pw.this.files_data),
        combined_hash=pw.apply(lambda h: "-".join(sorted(h)), pw.this.hashes)
    )
    
    # PATHWAY UDF - Run the debate (only when hash changes)
    debate_results = results.select(
        pw.this.combined_hash,
        summary=process_debate_data(
            pw.this.fundamentals,
            pw.this.market,
            pw.this.news,
            pw.this.sentiment
        ),
        timestamp=pw.apply(lambda x: datetime.now().isoformat(), pw.this.combined_hash)
    )
    
    # PATHWAY OUTPUT - Write results using deduplicate to prevent re-runs on same content
    deduplicated = debate_results.deduplicate(
    value=pw.this.combined_hash,
    acceptor=lambda new, old: (old is None) or (new != old),
    name="debate_results_dedup"
)
    
    pw.io.jsonlines.write(
        deduplicated,
        f"{OUTPUT_FOLDER}/debate_results.jsonlines"
    )
    
    print("\n💡 Pathway pipeline configured:")
    print("  ├─ Input: Streaming file reader (3s polling)")
    print("  ├─ Transform: Content hashing for change detection")
    print("  ├─ Group: Aggregate all 4 files together")
    print("  ├─ Filter: Only process when all 4 files present")
    print("  ├─ Process: Run debate via UDF")
    print("  ├─ Deduplicate: Skip if content unchanged")
    print("  └─ Output: Write results to JSONL")
    print("\n🔄 Edit any .md file to trigger debate")
    print("🛑 Press Ctrl+C to stop\n")
    
    # RUN PATHWAY
    pw.run()