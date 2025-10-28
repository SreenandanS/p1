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

BEAR_SUMMARIZER_PROMPT = """You are an expert financial analyst tasked with creating a comprehensive Bear Case Summary Report. Your role is to synthesize and validate all bearish arguments presented during the debate.

Your report should include:

1. *Executive Summary of Bear Position*
   - Concise overview of the primary bearish thesis
   - Key risk factors identified

2. *Detailed Risk Analysis*
   - Market and competitive risks
   - Financial vulnerabilities
   - Operational challenges
   - Regulatory or macroeconomic threats
   
3. *Validation of Bear Arguments*
   - Which bearish points were most compelling and why
   - Evidence and data supporting the bear case
   - Logical consistency of the arguments
   
4. *Counterpoint Assessment*
   - How well did bear arguments address bull counterpoints
   - Remaining weaknesses in the bear position
   
5. *Key Takeaways*
   - Most significant concerns for investors
   - Critical factors to monitor

Do NOT provide any investment recommendation (BUY/SELL/HOLD). Focus purely on analyzing and documenting the bear case."""

BULL_SUMMARIZER_PROMPT = """You are an expert financial analyst tasked with creating a comprehensive Bull Case Summary Report. Your role is to synthesize and validate all bullish arguments presented during the debate.

Your report should include:

1. *Executive Summary of Bull Position*
   - Concise overview of the primary bullish thesis
   - Key growth drivers identified

2. *Detailed Growth Analysis*
   - Market opportunities and competitive advantages
   - Financial strengths and momentum
   - Operational excellence and innovation
   - Favorable market trends and catalysts
   
3. *Validation of Bull Arguments*
   - Which bullish points were most compelling and why
   - Evidence and data supporting the bull case
   - Logical consistency of the arguments
   
4. *Counterpoint Assessment*
   - How well did bull arguments address bear counterpoints
   - Remaining weaknesses in the bull position
   
5. *Key Takeaways*
   - Most significant opportunities for investors
   - Critical success factors to monitor

Do NOT provide any investment recommendation (BUY/SELL/HOLD). Focus purely on analyzing and documenting the bull case."""

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
def compute_hash(data: str) -> str:
    """Compute hash of file content to detect actual changes"""
    import hashlib
    return hashlib.md5(data.encode()).hexdigest() if data else ""

@pw.udf
def has_four_files(filenames_tuple) -> bool:
    """Check if we have all 4 required files"""
    required = {"fundamentals_report.md", "market_report.md", "news_report.md", "sentiment_report.md"}
    return required.issubset(set(filenames_tuple)) if filenames_tuple else False

@pw.udf
def get_file_by_name(filenames_tuple, data_tuple, target_filename: str) -> str:
    """Extract specific file content by filename"""
    if not filenames_tuple or not data_tuple:
        return ""
    try:
        file_dict = dict(zip(filenames_tuple, data_tuple))
        return file_dict.get(target_filename, "")
    except:
        return ""

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
        
        print("\n✅ Debate rounds completed!")
        
        # Create debate transcript
        print("📝 Creating debate transcript...")
        debate_text = "\n\n".join([
            f"Round {item['round']}:\nBull: {item['bull']}\nBear: {item['bear']}" 
            for item in history
        ])
        print(f"✅ Transcript created ({len(debate_text)} characters)")
        
        # Generate bear summary
        print("\n📊 Generating bear case summary...")
        
        bear_summary_prompt = [
            {"role": "system", "content": BEAR_SUMMARIZER_PROMPT},
            {"role": "user", "content": f"""Review the following debate and create a comprehensive Bear Case Summary Report.

Available Research:
Market Research Report: {market[:300]}...
Social Media Sentiment Report: {sentiment[:300]}...
World Affairs News: {news[:300]}...
Company Fundamentals Report: {fundamentals[:300]}...

Debate Transcript:
{debate_text}

Create a detailed bear case summary following the structured format provided in your instructions."""}
        ]
        
        bear_summary_table = pw.debug.table_from_pandas(pd.DataFrame({"messages": [bear_summary_prompt]}))
        bear_summary_response = bear_summary_table.select(summary=chat_model(pw.this.messages))
        bear_summary_result = pw.debug.table_to_pandas(bear_summary_response)
        bear_summary = bear_summary_result["summary"].iloc[0] if not bear_summary_result.empty else "Bear summary completed"
        print(f"✅ Bear summary generated ({len(bear_summary)} characters)")
        
        print("\n📊 Generating bull case summary...")
        
        bull_summary_prompt = [
            {"role": "system", "content": BULL_SUMMARIZER_PROMPT},
            {"role": "user", "content": f"""Review the following debate and create a comprehensive Bull Case Summary Report.

Available Research:
Market Research Report: {market[:300]}...
Social Media Sentiment Report: {sentiment[:300]}...
World Affairs News: {news[:300]}...
Company Fundamentals Report: {fundamentals[:300]}...

Debate Transcript:
{debate_text}

Create a detailed bull case summary following the structured format provided in your instructions."""}
        ]
        
        bull_summary_table = pw.debug.table_from_pandas(pd.DataFrame({"messages": [bull_summary_prompt]}))
        bull_summary_response = bull_summary_table.select(summary=chat_model(pw.this.messages))
        bull_summary_result = pw.debug.table_to_pandas(bull_summary_response)
        bull_summary = bull_summary_result["summary"].iloc[0] if not bull_summary_result.empty else "Bull summary completed"
        print(f"✅ Bull summary generated ({len(bull_summary)} characters)")
        
        # Save separate files
        clean_timestamp = timestamp.replace(':', '-').replace(' ', '_')
        
        print(f"\n💾 Saving files with timestamp: {clean_timestamp}")
        
        # Save debate transcript
        debate_path = Path(OUTPUT_FOLDER) / f"debate.md"
        debate_content = f"""# Stock Analysis Debate Transcript
*Generated:* {timestamp}

---

{debate_text}
"""
        debate_path.write_text(debate_content)
        print(f"✅ Debate saved: {debate_path.absolute()}")
        
        # Save bear report
        bear_path = Path(OUTPUT_FOLDER) / f"bear_report.md"
        bear_content = f"""# Bear Case Summary Report
*Generated:* {timestamp}

---

{bear_summary}
"""
        bear_path.write_text(bear_content)
        print(f"✅ Bear report saved: {bear_path.absolute()}")
        
        # Save bull report
        bull_path = Path(OUTPUT_FOLDER) / f"bull_report.md"
        bull_content = f"""# Bull Case Summary Report
*Generated:* {timestamp}

---

{bull_summary}
"""
        bull_path.write_text(bull_content)
        print(f"✅ Bull report saved: {bull_path.absolute()}")
        
        print(f"\n🎉 All 3 files generated successfully!")
        print(f"{'='*80}\n")
        
        return f"Reports generated: {debate_path.name}, {bear_path.name}, {bull_path.name}"
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

# ----------------------------
# MAIN PATHWAY PIPELINE
# ----------------------------
if _name_ == "_main_":
    print("="*80)
    print("🚀 PATHWAY STOCK DEBATE SYSTEM - FILENAME-BASED")
    print("="*80)
    
    # Check for required files
    required_files = ["fundamentals_report.md", "market_report.md", "news_report.md", "sentiment_report.md"]
    txt_files = [Path(DATA_FOLDER) / f for f in required_files]
    missing = [f.name for f in txt_files if not f.exists()]
    
    if missing:
        raise ValueError(
            f"Missing required files in '{DATA_FOLDER}' folder: {', '.join(missing)}\n"
            f"Please ensure you have:\n"
            "1. fundamentals_report.md\n"
            "2. market_report.md\n"
            "3. news_report.md\n"
            "4. sentiment_report.md"
        )
    
    print(f"📁 Found all required report files")
    print(f"📊 Setting up Pathway streaming pipeline")
    print("="*80)
    
    # PATHWAY STREAMING INPUT - Read each file separately and tag with filename
    fundamentals_stream = pw.io.fs.read(
        path=f"{DATA_FOLDER}/fundamentals_report.md",
        format="plaintext",
        mode="streaming",
        autocommit_duration_ms=3000
    ).select(
        data=pw.this.data,
        filename=pw.apply(lambda x: "fundamentals_report.md", pw.this.data),
        content_hash=compute_hash(pw.this.data)
    )
    
    market_stream = pw.io.fs.read(
        path=f"{DATA_FOLDER}/market_report.md",
        format="plaintext",
        mode="streaming",
        autocommit_duration_ms=3000
    ).select(
        data=pw.this.data,
        filename=pw.apply(lambda x: "market_report.md", pw.this.data),
        content_hash=compute_hash(pw.this.data)
    )
    
    news_stream = pw.io.fs.read(
        path=f"{DATA_FOLDER}/news_report.md",
        format="plaintext",
        mode="streaming",
        autocommit_duration_ms=3000
    ).select(
        data=pw.this.data,
        filename=pw.apply(lambda x: "news_report.md", pw.this.data),
        content_hash=compute_hash(pw.this.data)
    )
    
    sentiment_stream = pw.io.fs.read(
        path=f"{DATA_FOLDER}/sentiment_report.md",
        format="plaintext",
        mode="streaming",
        autocommit_duration_ms=3000
    ).select(
        data=pw.this.data,
        filename=pw.apply(lambda x: "sentiment_report.md", pw.this.data),
        content_hash=compute_hash(pw.this.data)
    )
    
    # Concatenate all streams
    files = pw.Table.concat_reindex(fundamentals_stream, market_stream, news_stream, sentiment_stream)
    
    # PATHWAY GROUPBY - Group all files together by a constant key
    files_with_key = files.select(
        pw.this.data,
        pw.this.filename,
        pw.this.content_hash,
        group_key=pw.apply(lambda x: "all_files", pw.this.data)
    )
    
    # PATHWAY REDUCE - Collect all file data together with filenames
    grouped = files_with_key.groupby(pw.this.group_key).reduce(
        pw.this.group_key,
        filenames=pw.reducers.tuple(pw.this.filename),
        files_data=pw.reducers.tuple(pw.this.data),
        hashes=pw.reducers.tuple(pw.this.content_hash)
    )
    
    # PATHWAY FILTER - Only process when we have all 4 required files
    grouped = grouped.filter(has_four_files(pw.this.filenames))
    
    # PATHWAY SELECT - Extract individual reports by filename
    results = grouped.select(
        fundamentals=get_file_by_name(pw.this.filenames, pw.this.files_data, "fundamentals_report.md"),
        market=get_file_by_name(pw.this.filenames, pw.this.files_data, "market_report.md"),
        news=get_file_by_name(pw.this.filenames, pw.this.files_data, "news_report.md"),
        sentiment=get_file_by_name(pw.this.filenames, pw.this.files_data, "sentiment_report.md"),
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
    # Group by hash and keep only the first occurrence
    deduplicated = debate_results.groupby(pw.this.combined_hash).reduce(
        pw.this.combined_hash,
        summary=pw.reducers.argmin(pw.this.timestamp, pw.this.summary),
        timestamp=pw.reducers.min(pw.this.timestamp)
    )
    
    pw.io.jsonlines.write(
        deduplicated,
        f"{OUTPUT_FOLDER}/debate_results.jsonlines"
    )
    
    print("\n💡 Pathway pipeline configured:")
    print("  ├─ Input: 4 separate file streams (filename-tagged)")
    print("  ├─ Transform: Content hashing for change detection")
    print("  ├─ Concat: Merge all streams with reindexing")
    print("  ├─ Group: Aggregate files by constant key")
    print("  ├─ Filter: Only process when all 4 files present")
    print("  ├─ Extract: Get files by explicit name matching")
    print("  ├─ Process: Run debate via UDF")
    print("  ├─ Deduplicate: Skip if content unchanged")
    print("  └─ Output: Write results to JSONL")
    print("\n🔄 Edit any required .md file to trigger debate")
    print("🛑 Press Ctrl+C to stop\n")
    
    # RUN PATHWAY
    pw.run()
