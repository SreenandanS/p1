import os
from pathlib import Path
from dotenv import load_dotenv
import openai

# ----------------------------
# CONFIGURATION
# ----------------------------
DATA_FOLDER = "data-source"
OUTPUT_FOLDER = "./final_reports"
N_ROUNDS = 3  
load_dotenv()

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# AGENT CLASS DEFINITIONS
# ----------------------------
class DebateAgent:
    """Base class for debate agents using OpenAI API"""
    
    def __init__(self, system_prompt: str, role: str, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.system_prompt = system_prompt
        self.role = role
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.memory = []
    
    def generate_response(self, context: str, opponent_message: str, round_num: int) -> str:
        """Generate a response based on context and opponent's message"""
        
        # Build the full prompt with system instructions and context
        full_prompt = f"""{context}

Round {round_num}:
Opponent ({self.get_opponent_role()}) said: {opponent_message}

Your turn to argue as the {self.role}. Provide a compelling, well-reasoned response that directly engages with the opponent's points.
"""
        
        # Generate response using OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature
        )
        
        response_text = response.choices[0].message.content
        
        # Store in memory
        self.memory.append({
            "round": round_num,
            "opponent": opponent_message,
            "response": response_text
        })
        
        return response_text
    
    def get_opponent_role(self) -> str:
        return "Bull" if self.role == "Bear" else "Bear"


class BearAgent(DebateAgent):
    """Bear analyst agent - argues against investment"""
    
    def __init__(self):
        system_prompt = """You are a Bear Analyst making the case against investing in the stock. Your goal is to present a well-reasoned argument emphasizing risks, challenges, and negative indicators.

Key points to focus on:
- Risks and Challenges: Highlight factors like market saturation, financial instability, or macroeconomic threats
- Competitive Weaknesses: Emphasize vulnerabilities such as weaker market positioning or threats from competitors  
- Negative Indicators: Use evidence from financial data, market trends, or recent adverse news
- Bull Counterpoints: Critically analyze the bull argument with specific data and sound reasoning
- Engagement: Present your argument conversationally, directly engaging with the bull analyst's points

Be critical, thorough, and evidence-based in your analysis."""
        
        super().__init__(system_prompt, "Bear")


class BullAgent(DebateAgent):
    """Bull analyst agent - argues for investment"""
    
    def __init__(self):
        system_prompt = """You are a Bull Analyst advocating for investing in the stock. Your task is to build a strong, evidence-based case emphasizing growth potential, competitive advantages, and positive market indicators.

Key points to focus on:
- Growth Potential: Highlight market opportunities, revenue projections, and scalability
- Competitive Advantages: Emphasize unique products, strong branding, or dominant market positioning
- Positive Indicators: Use financial health, industry trends, and recent positive news as evidence
- Bear Counterpoints: Critically analyze the bear argument with data and sound reasoning
- Engagement: Present your argument conversationally, engaging directly with the bear analyst's points

Be optimistic, thorough, and evidence-based in your analysis."""
        
        super().__init__(system_prompt, "Bull")


class SummarizerAgent:
    """Agent to summarize debate and provide final recommendation"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def summarize_debate(self, debate_history: list, reports_context: str) -> str:
        """Summarize the entire debate and provide BUY/HOLD/SELL recommendation"""
        
        # Compile debate transcript
        debate_text = "\n\n".join([
            f"Round {i+1}:\nBull: {bull}\nBear: {bear}" 
            for i, (bull, bear) in enumerate(debate_history)
        ])
        
        prompt = f"""You are an objective financial analyst. Review the following debate between Bull and Bear analysts and provide a comprehensive summary with a clear recommendation.

Available Research:
{reports_context}

Debate Transcript:
{debate_text}

Provide:
1. A balanced summary of key arguments from both sides
2. An assessment of the strongest points made
3. A clear final recommendation: BUY, HOLD, or SELL
4. Reasoning for your recommendation based on the debate and available data

Be objective and consider both perspectives before concluding."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an objective financial analyst who summarizes debates and provides clear investment recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )
        
        return response.choices[0].message.content


# ----------------------------
# DEBATE FUNCTION
# ----------------------------
def run_debate(reports, n_rounds=N_ROUNDS):
    """Run the debate between Bull and Bear agents"""
    
    # Unpack the four reports
    fundamentals_report, market_report, news_report, sentiment_report = reports
    
    # Build base context from all reports
    base_context = f"""
Market Research Report:
{market_report}

Social Media Sentiment Report:
{sentiment_report}

World Affairs News:
{news_report}

Company Fundamentals Report:
{fundamentals_report}
"""
    
    print("🎭 Starting Stock Analysis Debate...\n")
    
    # Initialize agents
    bull_agent = BullAgent()
    bear_agent = BearAgent()
    
    # Initialize debate history
    history = []
    bear_message = "Let's begin the analysis. I believe there are significant risks that investors should be aware of."
    
    # Run debate rounds
    for round_num in range(1, n_rounds + 1):
        print(f"📍 Round {round_num}")
        print("-" * 50)
        
        # Bull's turn
        print("🐂 Bull Analyst thinking...")
        bull_reply = bull_agent.generate_response(base_context, bear_message, round_num)
        print(f"Bull: {bull_reply[:200]}...\n")
        
        # Bear's turn
        print("🐻 Bear Analyst thinking...")
        bear_message = bear_agent.generate_response(base_context, bull_reply, round_num)
        print(f"Bear: {bear_message[:200]}...\n")
        
        # Store round
        history.append((bull_reply, bear_message))
    
    # Generate final summary
    print("📊 Generating final summary and recommendation...")
    summarizer = SummarizerAgent()
    summary = summarizer.summarize_debate(history, base_context)
    
    return summary


# ----------------------------
# MAIN LOGIC
# ----------------------------
def main():
    """Main execution function"""
    
    # Check for markdown files in data-source folder
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
    
    print(f"📁 Loading {len(txt_files)} report files...")
    
    # Read the first 4 markdown files
    reports = [Path(f).read_text() for f in txt_files[:4]]
    
    # Run the debate
    final_summary = run_debate(reports, N_ROUNDS)
    
    # Save output
    output_path = Path(OUTPUT_FOLDER) / "final_summary.txt"
    output_path.write_text(final_summary)
    
    print(f"\n✅ Final report saved at: {output_path}")
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(final_summary)


if __name__ == "__main__":
    main()